# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""this module provides the TAP class"""

import sys
from timeit import default_timer as time
import numpy as np
import cvxpy as cvx
from utils.logger import logger
from utils.configuration_parser import ConfigParser
from utils.common import ConstantParameters
from utils.generator import Generator
from task_allocation_problem.scenario_manager import ScenarioManager


class TaskAllocationProblem:
    """class to construct the task allocation problem"""

    def __init__(self) -> None:
        self.config: ConfigParser = ConfigParser()
        self.problem: cvx.Problem
        self.objective: cvx.Minimize
        self.constraints: list[cvx.Constraint] = []
        self.variables: dict[str, cvx.Variable | cvx.Expression] = {}
        self.constants: ConstantParameters
        self.scenario_manager: ScenarioManager = ScenarioManager(
            self.config, self.variables
        )

    def import_configurations(self, config_path: str) -> None:
        """import configuration to construct the task allocation problem"""
        logger.info("Start importing configuration files.")
        self.config.parse(config_path)
        logger.info("Finished importing configuration files.")

    def define_objectives_and_constraints(self) -> None:
        """define all objective and constraints based on the activated metrics"""

        self.scenario_manager.create_scenarios()
        self.objective = cvx.Minimize(self.scenario_manager.objective)
        self.constraints = self.scenario_manager.constraints

    def solve(self, verbose: bool = True):
        """solve the constructed problem"""
        # check if the problem is linear or quadratic
        solver: str = "CBC" if self.scenario_manager.is_problem_linear() else "SCIP"
        logger.info("Using solver: %s", solver)

        self.problem = cvx.Problem(self.objective, self.constraints)
        start_t = time()
        self.problem.solve(solver=solver, verbose=verbose)
        end_t = time()
        logger.info("time used (cvxpys reductions & solving): %0.4fs", end_t - start_t)
        logger.info("Problem status: %s", self.problem.status)
        logger.info("Objective value from solution: %s", self.problem.value)

    def generate(self, output_dir: str) -> None:
        """generate output files for the obtained orchestration"""
        generator = Generator(
            output_dir=output_dir,
            problem=self.problem,
            scenario_manager=self.scenario_manager,
        )
        generator.generate_ankaios_manifest(self.config)

    def print_solution(self) -> None:
        """log the obtained orchestration"""
        if self.problem.status != "optimal":
            logger.error("Problem could not be solved, cannot print a solution.")
            sys.exit(1)

        variables: dict = {
            key: value.value for key, value in self.problem.var_dict.items()
        }

        dummy_matrix: np.ndarray = np.zeros(
            next(
                (value for key, value in variables.items() if key.startswith("X_M"))
            ).shape
        )
        for scenario in self.config.scenarios.get_operation_scenarios():
            orchestration = (
                variables[f"X_M_{scenario}"]
                + variables.get(f"X_H_{scenario}", dummy_matrix) * 2
                + variables.get(f"X_P_{scenario}", dummy_matrix) * 3
                + variables.get(f"X_I_{scenario}", dummy_matrix) * 4
            )

            for (i, j), value in np.ndenumerate(orchestration):
                orchestration[i, j] = round(value)

            logger.info(
                "Generated orchestration for scenario %s is: \n %s",
                scenario,
                orchestration,
            )

            if self.config.scenarios.has_failure_scenarios:
                for failures in self.scenario_manager.failure_scenario_managers[
                    f"FSM_{scenario}"
                ].failure_graph:
                    current_failures = []
                    current_failures_string = ""

                    for f in failures:
                        current_failures.append(f)
                        current_failures_string = "_".join(
                            str(x) for x in current_failures
                        )

                        orchestration_f = (
                            variables[f"X_M_{scenario}_{current_failures_string}"]
                            + variables[f"X_H_{scenario}_{current_failures_string}"] * 2
                            + variables[f"X_P_{scenario}_{current_failures_string}"] * 3
                            + variables[f"X_I_{scenario}_{current_failures_string}"] * 4
                        )
                        logger.info(
                            "Nodes failed: %s \n Generated orchestration: \n %s",
                            current_failures,
                            orchestration_f,
                        )
