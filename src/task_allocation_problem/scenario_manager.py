# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""this module provides the the scenario manager class"""

from itertools import combinations
from copy import deepcopy
import numpy as np
import cvxpy as cvx

from utils.logger import logger
from utils.configuration_parser import ConfigParser
from utils.common import ConstantParameters
from task_allocation_problem.metrics import get_metric, get_scenario_metric, Metric


class ScenarioManager:
    """
    class to construct all relevant scenarios
    and corresponding variables / constraints
    """

    def __init__(
        self, config: ConfigParser, variables: dict[str, cvx.Variable | cvx.Expression]
    ) -> None:
        self.config: ConfigParser = config
        self.objective: cvx.Expression
        self.constraints: list[cvx.Constraint] = []
        self.metric_dict: dict[str, Metric] = {}
        self.variables: dict[str, cvx.Variable | cvx.Expression] = variables
        self.constants: ConstantParameters
        self.failure_scenario_managers: dict[str, FailureScenarioManager] = {}

    def create_scenarios(self) -> None:
        """create all operation and failure scenarios"""

        objective_list: list[cvx.Expression] = []
        weight_list: list[float] = []
        scenarios = self.config.scenarios.get_operation_scenarios()

        self._add_constants()
        weights: list = []
        objectives: list = []
        weighted_objective: list[cvx.Expression] = []

        for scenario in scenarios:
            # create variables, constraints, objectives for failure scenarios
            if self.config.scenarios.has_failure_scenarios:
                self.failure_scenario_managers[f"FSM_{scenario}"] = (
                    FailureScenarioManager(
                        scenario=scenario,
                        variables=self.variables,
                        constants=self.constants,
                        config=self.config,
                    )
                )
                self.failure_scenario_managers[f"FSM_{scenario}"].create_scenarios()

            else:
                # create variables, constraints, objectives for operation scenarios
                self.__add_variables_for_scenario(scenario)
                self.__add_constraints_for_scenario(scenario)

            objective_list, weight_list = self._import_metrics(scenario)

            if self.config.scenarios.has_failure_scenarios:
                for con in self.failure_scenario_managers[
                    f"FSM_{scenario}"
                ].constraints:
                    self.constraints.append(con)

                weight_list.append(1)
                objective_list.append(
                    self.failure_scenario_managers[f"FSM_{scenario}"].objective
                )

            weights.append(cvx.vstack(weight_list))
            objectives.append(cvx.vstack(objective_list))
            weighted_objective.append(
                cvx.sum(cvx.multiply(weights[-1], objectives[-1]))
            )

        self.objective = cvx.sum(weighted_objective)

    def is_problem_linear(self) -> bool:
        """check if all used metrics are linear"""
        metrics: list[bool] = [
            metric.is_linear for _, metric in self.metric_dict.items()
        ]
        return all(metrics)

    def _add_constants(self) -> None:
        """add constants used for problem definition"""
        self.constants = ConstantParameters(
            n_nodes=self.config.nodes.number_of_nodes,
            n_services=self.config.services.number_of_services,
            ones_vector_n_nodes=np.ones(
                shape=(1, self.config.nodes.number_of_nodes)
            ),  # helper for several calculations
        )

    def _import_metrics(
        self, scenario: str
    ) -> tuple[list[cvx.Expression], list[float]]:
        """import active metrics to construct the problem"""
        objective_list: list[cvx.Expression] = []
        weight_list: list[float] = []

        for metric in self.config.metrics.config:
            if self.config.metrics.config[metric]["is_active"] is True:
                logger.debug("Adding metric: %s", metric)
                # get_metric(metric) returns the class with name metric
                # pass all possible parameters to init of class object.
                # the metrics must pick the parameters they need and ignore the rest.
                self.metric_dict[f"{metric}_{scenario}"] = get_metric(metric)(
                    variables=self.variables,
                    scenario=scenario,
                    config=self.config,
                    constants=self.constants,
                )

                for con in self.metric_dict[f"{metric}_{scenario}"].get_constraints():
                    self.constraints.append(con)

                weight_list.append(self.config.metrics.config[metric]["weight"])
                objective_list.append(
                    self.metric_dict[f"{metric}_{scenario}"].get_objective()
                )

        logger.info("Used metrics are: %s", ", ".join(self.metric_dict))
        # set dummy values if no metrics are used
        if not objective_list:
            objective_list = [cvx.Constant(0)]
        if not weight_list:
            weight_list = [0]
        return objective_list, weight_list

    def __add_variables_for_scenario(self, scenario: str) -> None:
        """add decision variables for each failure scenarios"""
        variables: list[str] = [f"X_M_{scenario}"]
        for var in variables:
            if var not in self.variables:
                self.variables[var] = cvx.Variable(
                    (self.constants.n_nodes, self.constants.n_services),
                    boolean=True,
                    name=var,
                )

        if f"X_{scenario}" not in self.variables:
            self.variables[f"X_{scenario}"] = self.variables[f"X_M_{scenario}"]

    def __add_constraints_for_scenario(self, scenario: str) -> None:
        """add global constraints to the problem definition"""

        # each service that is active in the current scenario instance is
        # assigned only once. services that are not active, are not assigned
        active_services: np.ndarray = self.config.services.get_active_services(scenario)
        self.constraints.append(
            cvx.reshape(
                cvx.sum(self.variables[f"X_M_{scenario}"], axis=0),
                shape=(self.constants.n_services, 1),
            )
            == active_services
        )


class FailureScenarioManager(ScenarioManager):
    """
    class to construct all relevant failure scenarios
    and corresponding variables / constraints
    """

    def __init__(
        self,
        scenario: str,
        constants: ConstantParameters,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario
        self.constants = constants
        self.failure_graph: list[tuple[int, int]]

    def create_scenarios(self) -> None:
        number_of_failures = 0
        # constraints for operation scenario without failure
        self.__add_variables_for_scenario(scenario=f"{self.scenario}")
        self.__add_constraints_for_scenario(
            scenario=self.scenario, number_of_failures=number_of_failures
        )

        max_failures = np.max(self.config.services.get_service_fail_operational_level())
        logger.debug("Maximum fail_operational_level is %i", max_failures)

        failure_graph: list[tuple[int, int]] = list(
            combinations(range(self.constants.n_nodes), int(max_failures))  # type: ignore
        )
        logger.debug(
            "Obtained failure_graph for fail_operational_level: %s", failure_graph
        )

        node_failures = self.__create_operation_failure_graph_helper(
            failure_graph=failure_graph
        )

        obj: cvx.Expression = cvx.Constant(0)
        for failures in failure_graph:
            number_of_failures = 0
            previous_failures = []
            previous_failures_string = ""
            current_failures = []
            current_failures_string = ""

            for f in failures:
                number_of_failures = number_of_failures + 1
                current_failures.append(f)
                current_failures_string = "_".join(str(x) for x in current_failures)

                # TODO: use __add_variables and __add_global_constraints method
                # from TaskAllocationProblem without having circular imports.
                # Maybe create a new component in utils
                self.__add_variables_for_scenario(
                    scenario=f"{self.scenario}_{current_failures_string}"
                )

                # global constraints
                self.__add_constraints_for_scenario(
                    scenario=f"{self.scenario}_{current_failures_string}",
                    number_of_failures=number_of_failures,
                )

                # ensure that services are not assigned on failed nodes
                self.constraints.append(
                    self.variables[f"X_{self.scenario}_{current_failures_string}"]
                    <= node_failures[f"NF_{current_failures_string}_matrix"]
                )

                # ensure that a failed service is restarted on the node, where
                # its backup instance was assigned
                # X_M_i + X_M_{i-1}[f] * (X_H_{i-1} + X_P_{i-1}) <= 1
                # X_M_f_i = X_M_{i-1}[f] * (X_H_{i-1} + X_P_{i-1})
                # X_M_f_i <= (X_H_{i-1} + X_P_{i-1})
                # X_M_f_i <= X_M_{i-1}[f]
                # X_M_f_i >= X_M_{i-1}[f] + (X_H_{i-1} + X_P_{i-1}) - 1
                self.variables[f"X_M_{f}_{self.scenario}_{current_failures_string}"] = (
                    cvx.Variable(
                        (self.constants.n_nodes, self.constants.n_services),
                        boolean=True,
                        name=f"X_M_{f}_{self.scenario}_{current_failures_string}",
                    )
                )

                previous_failures_string_var: str = (
                    "_" + previous_failures_string
                    if previous_failures_string != ""
                    else ""
                )

                self.constraints.append(
                    self.variables[f"X_M_{f}_{self.scenario}_{current_failures_string}"]
                    <= self.variables[
                        f"X_H_{self.scenario}{previous_failures_string_var}"
                    ]
                    + self.variables[
                        f"X_P_{self.scenario}{previous_failures_string_var}"
                    ]
                    + self.variables[
                        f"X_I_{self.scenario}{previous_failures_string_var}"
                    ]
                )
                self.constraints.append(
                    self.variables[f"X_M_{f}_{self.scenario}_{current_failures_string}"]
                    <= cvx.multiply(
                        cvx.reshape(
                            self.variables[
                                f"X_M_{self.scenario}{previous_failures_string_var}"
                            ][f],
                            (1, self.constants.n_services),
                        ),
                        cvx.Constant(
                            np.ones((self.constants.n_nodes, self.constants.n_services))
                        ),
                    )
                )
                self.constraints.append(
                    self.variables[f"X_M_{f}_{self.scenario}_{current_failures_string}"]
                    >= cvx.multiply(
                        cvx.reshape(
                            self.variables[
                                f"X_M_{self.scenario}{previous_failures_string_var}"
                            ][f],
                            (1, self.constants.n_services),
                        ),
                        cvx.Constant(
                            np.ones((self.constants.n_nodes, self.constants.n_services))
                        ),
                    )
                    + self.variables[
                        f"X_H_{self.scenario}{previous_failures_string_var}"
                    ]
                    + self.variables[
                        f"X_P_{self.scenario}{previous_failures_string_var}"
                    ]
                    + self.variables[
                        f"X_I_{self.scenario}{previous_failures_string_var}"
                    ]
                    - 1
                )
                self.constraints.append(
                    self.variables[f"X_M_{self.scenario}_{current_failures_string}"]
                    >= self.variables[
                        f"X_M_{f}_{self.scenario}_{current_failures_string}"
                    ]
                )

                # add constraints of all activated metrics also to the failure scenario
                # TODO: add objectives of metrics also to problem?
                self._import_metrics(
                    scenario=f"{self.scenario}_{current_failures_string}"
                )
                # https://retis.sssup.it/~lipari/papers/mtbs_baruah_lipari.pdf equation 9

                ftti = get_scenario_metric("FaultToleranceTimeInterval")(
                    config=self.config,
                    scenario=self.scenario,
                    variables=self.variables,
                    constants=self.constants,
                    current_failure=current_failures_string,
                    previous_failure=previous_failures_string,
                )

                obj += ftti.get_objective()
                for con in ftti.get_constraints():
                    self.constraints.append(con)

                previous_failures.append(f)
                previous_failures_string = "_".join(str(x) for x in previous_failures)

        # normalize objective
        # number of combinations from binomial coefficient n! / (k! (n - k)!)
        # with n=N_NODES and k=N_NODES-1 -> results in N_NODES
        # total number of reconfiguration events is therefore N_NODES * (max_failures)
        self.objective = obj / (self.constants.n_nodes * max_failures)

        self.failure_graph = failure_graph

    def __create_operation_failure_graph_helper(
        self, failure_graph: list[tuple[int, int]]
    ) -> dict:
        """
        Creates matrices as cvx.Constants for each failure that can happen.
        The matrices hold a 1 for all healthy nodes in each scenario and a 0 for the failed ones.
        """
        node_failures = {}
        for failures in failure_graph:
            previous_failures = []
            previous_failures_string = ""
            node_failure = np.eye(self.constants.n_nodes)
            node_failure_matrix = np.ones(
                (self.constants.n_nodes, self.constants.n_services)
            )
            for f in failures:
                previous_failures.append(f)
                previous_failures_string = "_".join(str(x) for x in previous_failures)
                node_failure[f, f] = 0
                node_failure_matrix[f, :] = 0
                if f"NF_{previous_failures_string}" not in node_failures:
                    node_failures[f"NF_{previous_failures_string}"] = cvx.Constant(
                        deepcopy(node_failure)
                    )
                    node_failures[f"NF_{previous_failures_string}_matrix"] = (
                        cvx.Constant(deepcopy(node_failure_matrix))
                    )

        return node_failures

    def __add_variables_for_scenario(self, scenario: str) -> None:
        """add decision variables for each failure scenarios"""
        variables: list[str] = [
            f"X_M_{scenario}",
            f"X_H_{scenario}",
            f"X_P_{scenario}",
            f"X_I_{scenario}",
        ]
        for var in variables:
            if var not in self.variables:
                self.variables[var] = cvx.Variable(
                    (self.constants.n_nodes, self.constants.n_services),
                    boolean=True,
                    name=var,
                )

        if f"X_{scenario}" not in self.variables:
            self.variables[f"X_{scenario}"] = (
                self.variables[f"X_M_{scenario}"] + self.variables[f"X_H_{scenario}"]
            )

    def __add_constraints_for_scenario(
        self, scenario: str, number_of_failures: int
    ) -> None:
        """scenario based constraints for services"""

        active_services: np.ndarray = np.reshape(
            self.config.services.get_active_services(self.scenario),
            (1, self.constants.n_services),
        )
        fail_operational_level: np.ndarray = (
            self.config.services.get_service_fail_operational_level()
        )

        active_service_instances = np.multiply(
            np.maximum(fail_operational_level + 1 - number_of_failures, 0),
            active_services,
        )
        active_services_x_m: np.ndarray = np.array(
            active_service_instances > 0, dtype=int
        )

        # each service that is active in the current scenario has only one
        # active instance. services that are not active, are not assigned
        self.constraints.append(
            cvx.reshape(
                cvx.sum(
                    self.variables[f"X_M_{scenario}"],
                    axis=0,
                ),
                shape=(self.constants.n_services, 1),
            )
            == active_services_x_m
        )

        # each service that is active in the current scenario and has a
        # fail_operational_level >= 0, should have a hot-standy or passive
        # (cold-standby) or inactive instance assigned if the number of
        # failures is less than the fail_operational_level

        self.constraints.append(
            cvx.reshape(
                cvx.sum(
                    (
                        self.variables[f"X_M_{scenario}"]
                        + self.variables[f"X_H_{scenario}"]
                        + self.variables[f"X_P_{scenario}"]
                        + self.variables[f"X_I_{scenario}"]
                    ),
                    axis=0,
                ),
                shape=(self.constants.n_services, 1),
            )
            == active_service_instances
        )

        # hot-standby, passive (cold-standby) and inactive instances should not
        # be assigned on the same node

        self.constraints.append(
            (
                self.variables[f"X_M_{scenario}"]
                + self.variables[f"X_H_{scenario}"]
                + self.variables[f"X_P_{scenario}"]
                + self.variables[f"X_I_{scenario}"]
            )
            <= 1
        )
