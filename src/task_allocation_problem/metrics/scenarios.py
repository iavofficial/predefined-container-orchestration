# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""this module provides failure scenario related metrics"""

import numpy as np
import cvxpy as cvx
from utils.logger import logger
from utils.metric_helper import create_block_matrix
from task_allocation_problem.metrics.metric import Metric


class FaultToleranceTimeInterval(Metric):
    """
    METRIC FAULT TOLERANCE TIME INTERVAL BASED ON EDF

    Constaint: sum(WC_FTTI_i) <= FTTI_i --> worst case FTTI should not exceed FTTI_i
    Objective: min(sum(WC_FTTI_i)) --> minimize the total reconfiguration time

    WC_FTTI_i depends on the different possible state changes:
    1. state_{i-1} == state_i: WC_FTTI_i = 0
    2. inactive --> active: WC_FTTI_i = OrchestrationTime + ContainerPullTime + ContainerCreateTime + AppInitTime
    3. passive --> active: WC_FTTI_i = OrchestrationTime + AppInitTime
    4. hot-standby --> active: WC_FTTI_i = OrchestrationTime

    """

    def __init__(self, current_failure: str, previous_failure: str, **kwargs):
        self.current_failure = current_failure
        self.previous_failure = previous_failure

        super().__init__(**kwargs)

    def _apply(self) -> None:
        self.is_linear = True

        current_scenario: str = (
            f"{self.scenario}{(
            "_" + self.current_failure if self.current_failure != "" else ""
        )}"
        )
        previous_scenario: str = (
            f"{self.scenario}{(
            "_" + self.previous_failure if self.previous_failure != "" else ""
        )}"
        )
        # determine affected services as the diff between current orchestration
        # and previous orchestration. If a service is shifted from one node to
        # another, this will result in to differences in difference of orchestrations
        # --> max((X_i - X_{i-1}), 0)
        # however, max(X, 0) cannot be used in equality constraints because
        # max(X, 0) <= is non-convex and violated DCP ruleset
        # https://scicomp.stackexchange.com/questions/40904/why-is-a-elementwise-max-not-dcp

        self.variables[f"X_M_diff_{current_scenario}"] = cvx.Variable(
            (self.constants.n_nodes, self.constants.n_services),
            boolean=True,
            name=f"X_M_diff_{current_scenario}",
        )

        # model X_diff using boolean AND NOT logic:
        # if X_i == 1 and not X_{i-1} == 1 --> 1 else 0
        # https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators
        for i in range(self.constants.n_nodes):
            for j in range(self.constants.n_services):
                # X_diff <= X_i :
                self.constraints.append(
                    self.variables[f"X_M_diff_{current_scenario}"][i][j]
                    <= self.variables[f"X_M_{current_scenario}"][i][j]
                )
                # X_diff <= 1-X_{i-1}
                self.constraints.append(
                    self.variables[f"X_M_diff_{current_scenario}"][i][j]
                    <= 1 - self.variables[f"X_M_{previous_scenario}"][i][j]
                )
                # X_diff >= X_i + (1 - X_{i-1}) - 1
                self.constraints.append(
                    self.variables[f"X_M_diff_{current_scenario}"][i][j]
                    >= self.variables[f"X_M_{current_scenario}"][i][j]
                    + (1 - self.variables[f"X_M_{previous_scenario}"][i][j])
                    - 1
                )

        # create variables for started instances
        # X_M_H --> hot-standby instances that become active
        # X_M_P --> passive instances that become active
        # X_M_I --> inactive instances that become active
        diff_vars: list[tuple[str, str]] = [
            (f"X_H_{previous_scenario}", f"X_M_H_{current_scenario}"),
            (f"X_P_{previous_scenario}", f"X_M_P_{current_scenario}"),
            (f"X_I_{previous_scenario}", f"X_M_I_{current_scenario}"),
        ]
        for var2_name, name in diff_vars:
            self.__hadamard_product_of_variables(
                var1=self.variables[f"X_M_diff_{current_scenario}"],
                var2=self.variables[var2_name],
                name=name,
            )

        # OrchestrationTime is assumed to be a constant because
        # time complexity at runtime for determining an orchestration for
        # predefined reconfiguration is reduced from O(m^n) to O(1)
        ORCHESTRATION_TIME = cvx.Constant(
            100 * np.ones((self.constants.n_nodes, self.constants.n_services))
        )  # in ms

        # ContainerPullTime is assumed to be zero, e.g. all container
        # images are already present on all nodes
        CONTAINER_PULL_TIME = cvx.Constant(
            np.zeros((self.constants.n_nodes, self.constants.n_services))
        )

        # ContainerCreateTime depends on the number of cpu cores, cpu
        # frequency as well as the number of services to be created.
        # The number of services is the sum of each row in X_diff.
        # However, we cannot multiply the sum with the matrix itself, as this
        # would violate DCP rules. Therefore, we need to linearize to quadratic terms
        # e.g. x_11 * (x_11 + x_12 + x_13 + x_14) and sum afterwards

        Y = cvx.Variable(
            (
                self.constants.n_nodes * self.constants.n_services,
                self.constants.n_nodes * self.constants.n_services,
            ),
            boolean=True,
            name=f"X_M_I_dot_X_M_I_{current_scenario}",
        )

        for i in range(self.constants.n_nodes * self.constants.n_services):
            for j in range(self.constants.n_nodes * self.constants.n_services):
                self.constraints.append(
                    Y[i][j]
                    <= cvx.vec(self.variables[f"X_M_I_{current_scenario}"], order="C")[
                        i
                    ]
                )
                self.constraints.append(
                    Y[i][j]
                    <= cvx.vec(self.variables[f"X_M_I_{current_scenario}"], order="C")[
                        j
                    ]
                )
                self.constraints.append(
                    Y[i][j]
                    >= (
                        cvx.vec(self.variables[f"X_M_I_{current_scenario}"], order="C")[
                            i
                        ]
                        + cvx.vec(
                            self.variables[f"X_M_I_{current_scenario}"], order="C"
                        )[j]
                        - 1
                    )
                )

        JO = create_block_matrix(
            np.ones((self.constants.n_services, self.constants.n_services)),
            np.zeros((self.constants.n_services, self.constants.n_services)),
            self.constants.n_nodes,
            self.constants.n_nodes,
        )

        # SERVICES_PER_NODE == X_M_diff
        SERVICES_PER_NODE = cvx.reshape(
            cvx.sum(cvx.multiply(JO, Y), axis=1),
            (self.constants.n_nodes, self.constants.n_services),
            order="C",
        )

        CONTAINER_CREATE_TIME_PER_NODE = self.__get_container_create_time(
            self.variables[f"X_M_I_{current_scenario}"],
            SERVICES_PER_NODE,
            self.config.nodes.get_cpu_cores(),
            self.config.nodes.get_cpu_frequencies(),
        )

        # AppInitTime is the initialization time of an application on
        # startup. All applications are started with EDF scheduler,
        # therefore it is assumed that the app used its assigned runtime
        # to perform the initialization. The InitRuntime can be greater than
        # the normal runtime, meaning that the initialization may take several
        # periods to finish.
        INIT_TO_RUNTIME_RATIO = cvx.Constant(
            np.ceil(
                self.config.services.get_cpu_service_normalized_instructions_init()
                / self.config.services.get_cpu_service_normalized_instructions()
            )
        )

        P = cvx.Constant(self.config.services.get_cpu_service_period())

        WC_APP_INIT_TIME = cvx.multiply(INIT_TO_RUNTIME_RATIO, P)
        WC_APP_INIT_TIME_PER_NODE = cvx.vstack(
            [WC_APP_INIT_TIME for _ in range(self.constants.n_nodes)]
        )

        # WC_FTTI = X_M_H * Orchestration_Time
        #         + X_M_P * (Orchestration_Time + App_Init_Time)
        #         + X_M_I * (Orchestration_Time + Container_Pull_Time + Container_Create_Time + App_Init_Time)

        WC_FTTI = cvx.max(
            cvx.multiply(
                self.variables[f"X_M_H_{current_scenario}"], ORCHESTRATION_TIME
            )
            + cvx.multiply(
                self.variables[f"X_M_P_{current_scenario}"],
                (ORCHESTRATION_TIME + WC_APP_INIT_TIME_PER_NODE),
            )
            + cvx.multiply(
                self.variables[f"X_M_I_{current_scenario}"],
                (ORCHESTRATION_TIME + CONTAINER_PULL_TIME + WC_APP_INIT_TIME_PER_NODE),
            )
            + CONTAINER_CREATE_TIME_PER_NODE,
            axis=0,
        )

        FTTI = cvx.reshape(
            cvx.Constant(self.config.services.get_service_ftti()),
            (self.constants.n_services,),
        )
        self.constraints.append(WC_FTTI <= FTTI)

        # minimize the sum of normalized WC_FTTI and
        # minimize the number of hot-standby and passive instances
        # TODO: normalize WC_FTTI/FTTI with number of services that get reconfigured
        WEIGHT_X_H: float = (
            0.6  # weight is added to favor passive instance instead of hot-standby
        )
        n_active_services: cvx.Constant = cvx.Constant(
            np.sum(
                np.reshape(
                    self.config.services.get_active_services(self.scenario),
                    (1, self.constants.n_services),
                )
            )
        )

        self.objective = (cvx.sum(WC_FTTI / FTTI)) / n_active_services + (
            cvx.sum(
                WEIGHT_X_H * self.variables[f"X_H_{previous_scenario}"]
                + (1 - WEIGHT_X_H) * self.variables[f"X_P_{previous_scenario}"]
            )
            / (WEIGHT_X_H * n_active_services)
        )

    def __hadamard_product_of_variables(
        self,
        var1: cvx.Variable | cvx.Expression,
        var2: cvx.Variable | cvx.Expression,
        name: str,
    ) -> None:
        """create hadamard product of two binary variables via constraints"""
        self.variables[name] = cvx.Variable(shape=var1.shape, boolean=True, name=name)

        for i in range(self.variables[name].shape[0]):
            for j in range(self.variables[name].shape[1]):
                # X <= var1 :
                self.constraints.append(self.variables[name][i][j] <= var1[i][j])
                # X <= var2
                self.constraints.append(self.variables[name][i][j] <= var2[i][j])
                # X >= var1 + var2 - 1
                self.constraints.append(
                    self.variables[name][i][j] >= var1[i][j] + var2[i][j] - 1
                )

    def __get_container_create_time(
        self,
        affected_services: cvx.Variable | cvx.Expression,
        services_per_node: cvx.Expression,
        n_cores: np.ndarray,
        cpu_frequency: list[float],
    ) -> cvx.Expression:
        """
        Calculates the time to create a container based on
        the number of containers, cpu cores and cpu frequency

        Holds a linear WC limit for starting any number of containers
        wc_create_time = c0 + c1 * x
        however, c0 should only be added if x !=0, therefore we multiple c0 with
        affected_services, which is 1 if a service gets reconfigured and 0 else
        """
        # coef: np.ndarray = np.array([0.26458333, 0.23797043]) # util=0
        coef: np.ndarray = np.array([0.52083333, 0.35147849])  # util=90

        # # f(x) = c_0 + c_1 * x
        wc_create_time: cvx.Expression = (
            cvx.multiply(cvx.Constant(coef[0]), affected_services)
            + cvx.multiply(cvx.Constant(coef[1]), services_per_node)
        ) * 1000  # s to ms
        # wc_create_time: cvx.Expression = cvx.multiply(
        #     cvx.Constant(50), services_per_node
        # )

        return wc_create_time


class ScnearioDifferenceQuad(Metric):
    """
    OPERATION FAILURE SCENARIOS

    Creates failure scenarios and assumes that each node can fail.
    To reduce complexity it is assumed that the order of failures does not matter.

    constraints:
        - each required service instance is assigned
        - service instances are not assigned on failed nodes
    objective:
        minimize the number of reconfigurations =
            difference between orchestrations sum(|X_{i-1} - X_i|)/2)^2)/(N_SERVICES^2)

    ToDo:
        - define in service config a number of failures that a service must survive
        - add constraints of activated metrics for current scenario
        - add constraint to ensure Fault Tolerance Time is not violated
        - change objective to minimize the reconfiguration time
    """

    def _apply(self):
        """dummy apply"""
        # Objective:
        #   minimize the number of reconfigurations = difference between orchestrations
        # max number of reconfigurations is N_SERVICES --> used for normalization
        # absolute difference of matrices holds two entries with 1 for each
        # reconfiguration (1 on the source node, 1 on the target node) --> division by 2
        # square the difference to make n times reconfiguration of 1 service better
        # than 1 time reconfiguration of n services
        # but the objective ((sum(|X_{i-1} - X_i|)/2)^2)/(N_SERVICES^2) is not linear
        # obj += (
        #     cvx.square(
        #         cvx.sum(
        #             cvx.abs(
        #                 self.variables[
        #                     f"X_{self.scenario}{("_" + previous_failures_string
        #                                 if previous_failures_string != "" else "")}"
        #                 ]
        #                 - self.variables[
        #                     f"X_{self.scenario}{("_" + current_failures_string
        #                                     if current_failures_string != "" else "")}"
        #                 ]
        #             )
        #         )
        #         / 2
        #     )
        #     / cvx.square(self.constants.n_services)
        # )

        # linear alternative:
        #   maximum(sum(|X_{i-1} - X_i|), 1) / (2 * N_SERVICES + N_NODES - 1)

        # obj += cvx.maximum(cvx.sum(cvx.abs(
        #   operation_failure_scenarios[f"X_{previous_failures_string}"]
        #   - operation_failure_scenarios[f"X_{current_failures_string}"])), 1)
        #   / (2 * N_SERVICES + N_NODES - 1)

        # normalize objective
        # number of combinations from binomial coefficient n! / (k! (n - k)!)
        # with n=N_NODES and k=N_NODES-1 -> results in N_NODES
        # total number of reconfiguration events is therefore N_NODES * (N_NODES-1)
