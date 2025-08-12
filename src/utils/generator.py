# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""module provides generators to export the obtained orchestrations"""

import os
import yaml
from yaml.representer import SafeRepresenter
import numpy as np
import cvxpy as cvx

from utils.logger import logger
from utils.configuration_parser import ConfigParser
from task_allocation_problem.scenario_manager import ScenarioManager


class literal_str(str):
    pass


def change_style(style: str, representer: yaml.ScalarNode):
    """returns a representer function to change the style of the yaml"""

    def new_representer(dumper: str, data) -> yaml.ScalarNode:
        scalar: yaml.ScalarNode = representer(dumper, data)
        scalar.style = style
        return scalar

    return new_representer


represent_literal_str = change_style("|", SafeRepresenter.represent_str)

yaml.SafeDumper.add_representer(literal_str, represent_literal_str)


class Generator:
    """class to generate output files"""

    def __init__(
        self, output_dir: str, problem: cvx.Problem, scenario_manager: ScenarioManager
    ) -> None:
        self.output_dir = output_dir
        self.problem = problem
        self.scenario_manager = scenario_manager
        self.vc_path: str = os.path.join(self.output_dir, "vc")
        self.ank_manifest_path: str = os.path.join(self.output_dir, "ankaios_manifest")
        self.__create_output_dir()

    def __create_output_dir(self) -> None:
        """create the directory path if it does not exist"""
        if not os.path.exists(self.output_dir):
            self.__create_dir(self.output_dir)

    def __create_dir(self, directory_path: str) -> None:
        """create a directory if it does not exist"""
        logger.info(
            "output_dir: %s does not exist. Creating directory.", directory_path
        )
        try:
            os.makedirs(directory_path)
        except OSError as e:
            logger.error("Creating directory failed with: %s", e)

    def __clean_dir(self, directory_path: str) -> None:
        """remove all files in a directory"""
        logger.info(
            "output_dir: %s does already exist. Cleaning directory.",
            directory_path,
        )
        for file in os.listdir(directory_path):
            file_path: str = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except OSError as e:
                logger.error("Failed to remove file: %s due to exception: %s", file, e)

    def __get_orchestration_for_scenario(
        self,
        variable_type: str,
    ) -> np.ndarray:
        """retrieve the orchestration for a scenario"""

        res = {}
        for var in self.problem.variables():
            if var.name() == variable_type:
                res[var.name()] = var.value

        orchestration: np.ndarray = res[variable_type]

        for (i, j), value in np.ndenumerate(orchestration):
            orchestration[i, j] = round(value)

        return orchestration

    def __get_service_allocation_on_node_for_scenario(
        self, variable_type: str, service: str, config: ConfigParser
    ) -> str:
        """retrieve the allocated node name for a service in a scenario"""

        orchestration: np.ndarray = self.__get_orchestration_for_scenario(variable_type)
        service_id: int = config.services.config[service]["id"]

        node_id: int = -1
        for (i, j), _ in np.ndenumerate(orchestration):
            if j == service_id and orchestration[i, j] == 1:
                node_id = i
                break

        node_name: str = self.__get_node_name_from_id(config, node_id)

        return node_name

    def __get_command_options(self, service_data: dict, node_data: dict) -> list[str]:
        """retrieve a list of command options for service"""
        command_options: list[str] = []

        cpu_runtime_us: int = int(
            float(service_data["policies"]["task_normalized_instructions"])
            / node_data["cpu"]["frequency"]
            * 1000**2
        )
        cpu_period_us: int = int(service_data["policies"]["task_period"]) * 1000
        # cpu: float = cpu_runtime_us / cpu_period_us

        # command_options.extend(["--cpus", str(cpu)])
        command_options.extend(
            [
                "--cpu-rt-runtime",
                str(cpu_runtime_us),
            ]
        )
        command_options.extend(
            [
                "--cpu-rt-period",
                str(cpu_period_us),
            ]
        )
        command_options.extend(["-m", str(service_data["policies"]["memory"]) + "M"])

        return command_options

    def __get_node_name_from_id(self, config: ConfigParser, node_id: int) -> str:
        """get node name from node id"""
        for node in config.nodes.config:
            if config.nodes.config[node]["id"] == node_id:
                return str(config.nodes.config[node]["name"])

        return ""

    def __prepare_dir(self, dir_path: str) -> None:
        """prepare output directory"""
        if not os.path.exists(dir_path):
            self.__create_dir(dir_path)
        else:
            self.__clean_dir(dir_path)

    def generate_ankaios_manifest(self, config: ConfigParser) -> None:
        """generate an ankaios manifest for each scenario"""
        self.__prepare_dir(self.ank_manifest_path)

        scenarios: list[tuple[str, str]] = [(s, s) for s in config.scenarios.scenarios]
        failure_scenarios: list[str] = []
        if config.scenarios.has_failure_scenarios:
            for scenario in config.scenarios.scenarios:
                for failures in self.scenario_manager.failure_scenario_managers[
                    f"FSM_{scenario}"
                ].failure_graph:
                    current_failures = []
                    current_failures_string = ""

                    for fail in failures:
                        current_failures.append(fail)
                        current_failures_string = "_".join(
                            str(x) for x in current_failures
                        )
                        scenarios.append(
                            (scenario, f"{scenario}_{current_failures_string}")
                        )

                        failure_scenarios.append(
                            f"{scenario}_{current_failures_string}"
                        )

        orchestrations: dict[str, dict] = {}
        for scenario, scenario_variable_name in scenarios:
            if scenario not in orchestrations:
                orchestrations[scenario] = {}
            with open(
                file=os.path.join(
                    self.ank_manifest_path, f"{scenario_variable_name}.yaml"
                ),
                mode="w",
                encoding="utf-8",
            ) as f:
                workloads: dict = {}
                for _, service_data in config.services.config.items():
                    if scenario == "Default" or scenario in service_data["scenarios"]:
                        variable_types: list[tuple[str, str]] = [
                            ("X_M", scenario_variable_name)
                        ]
                        is_failure_scenario: bool = False
                        # failed_node: str = ""
                        if config.scenarios.has_failure_scenarios:
                            variable_types.append(("X_H", scenario_variable_name))
                            variable_types.append(("X_P", scenario_variable_name))
                            if scenario_variable_name in failure_scenarios:
                                is_failure_scenario = True
                                # failed_node = (
                                #     self.__get_node_name_from_failure_scenario(
                                #         config, scenario_variable_name
                                #     )
                                # )

                        for prefix, variable in variable_types:
                            node_name = (
                                self.__get_service_allocation_on_node_for_scenario(
                                    f"{prefix}_{variable}",
                                    service_data["name"],
                                    config,
                                )
                            )
                            if node_name == "":
                                continue

                            node_data: dict
                            for _, node_data in config.nodes.config.items():
                                if node_data["name"] == node_name:
                                    break

                            restart_policy: str = "ALWAYS"
                            if not is_failure_scenario and prefix == "X_P":
                                restart_policy = "NEVER"

                            # check if service was reconfigured
                            is_reconfigured: bool = False
                            if (
                                is_failure_scenario
                                and orchestrations[scenario][
                                    f"{service_data["name"]}_{prefix}"
                                ]
                                != node_name
                            ):
                                is_reconfigured = True

                            if is_reconfigured or (
                                not is_reconfigured and prefix in ["X_H", "X_P"]
                            ):
                                prefix = "Backup"

                            if is_failure_scenario and not is_reconfigured:
                                continue

                            orchestrations[scenario][
                                f"{service_data["name"]}_{prefix}"
                            ] = node_name

                            workloads[f"{service_data["name"]}_{prefix}"] = {
                                "runtime": "podman",
                                "agent": node_name,
                                "restartPolicy": restart_policy,
                                "tags": [{"key": "owner", "value": "Ankaios"}],
                                "runtimeConfig": literal_str(
                                    (
                                        f"image: {service_data["image"]}\n"
                                        f"commandOptions: {self.__get_command_options(service_data, node_data)}\n"
                                    )
                                ),
                            }
                data = {"apiVersion": "v0.1", "workloads": workloads}

                yaml.safe_dump(data=data, stream=f, sort_keys=False)
