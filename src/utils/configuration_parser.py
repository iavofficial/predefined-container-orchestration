# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""module provides a parser for the configuration file"""

from pathlib import Path
import copy
import yaml
import numpy as np
from utils.logger import logger
from utils.schema_validation import validate_with_meta_schema
from utils.settings import META_SCHEMA_PATH


class NodeParser:
    """class to create a parser for node related information"""

    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.__add_enumerate_nodes()

    @property
    def number_of_nodes(self) -> int:
        """number of nodes in the config"""
        return len(self.config)

    def __add_enumerate_nodes(self) -> None:
        """numerate all nodes"""
        for ind, n in enumerate(self.config):
            self.config[n]["id"] = ind

    def get_cpu_cores(self) -> np.ndarray:
        """get array of cpu cores for each node"""
        n_cores = []
        for n in self.config:
            n_cores.append(self.config[n]["cpu"]["n_cores"])
        return np.array(n_cores, dtype=int)

    def get_cpu_frequencies(self) -> list[float]:
        """get list of cpu frequencies for each node"""
        freq = []
        for n in self.config:
            freq.append(float(self.config[n]["cpu"]["frequency"]))
        return freq

    def get_net_bandwidth(self) -> np.ndarray:
        """get array of network bandwidth for each node"""
        net = []
        for n in self.config:
            net.append(self.config[n]["network"])
        return np.array(net, dtype=int)

    def get_mem_ram(self) -> np.ndarray:
        """get array of memory for each node"""
        mem = []
        for n in self.config:
            mem.append(self.config[n]["memory"])
        return np.array(mem, dtype=int)


class ServiceParser:
    """class to create a parser for service related information"""

    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.__add_enumerate_services()

    @property
    def number_of_services(self) -> int:
        """the number of services"""
        return len(self.config)

    def __add_enumerate_services(self) -> None:
        """numerate all services"""
        for ind, s in enumerate(self.config):
            self.config[s]["id"] = ind

    @property
    def name_to_id_dict(self) -> dict:
        """dict to map service names to service ids"""
        name_to_id = {}
        for s in self.config:
            name_to_id[self.config[s]["name"]] = self.config[s]["id"]
        return name_to_id

    def get_cpu_service_normalized_instructions(self) -> np.ndarray:
        """get an array of normalized instructions policies for each service"""
        cpu = []
        for s in self.config:
            cpu.append(self.config[s]["policies"]["task_normalized_instructions"])
        return np.array(cpu, dtype=int).reshape((1, self.number_of_services))

    def get_cpu_service_normalized_instructions_init(self) -> np.ndarray:
        """get an array of normalized instructions policies of initialization for each service"""
        cpu = []
        for s in self.config:
            if "task_normalized_instructions_init" in self.config[s]["policies"]:
                cpu.append(
                    self.config[s]["policies"]["task_normalized_instructions_init"]
                )
            else:
                # if not defined init=runtime
                cpu.append(self.config[s]["policies"]["task_normalized_instructions"])
        return np.array(cpu, dtype=int).reshape((1, self.number_of_services))

    def get_cpu_service_period(self) -> np.ndarray:
        """get an array of service periods policies"""
        cpu = []
        for s in self.config:
            cpu.append(float(self.config[s]["policies"]["task_period"]))
        return np.array(cpu, dtype=float).reshape((1, self.number_of_services))

    def get_service_ftti(self) -> np.ndarray:
        """get an array of service ftti policies"""
        cpu = []
        for s in self.config:
            if "ftti" in self.config[s]["policies"]:
                cpu.append(float(self.config[s]["policies"]["ftti"]))
            else:
                # if not defined FTTI = 3*period
                cpu.append(float(self.config[s]["policies"]["task_period"]) * 3)
        return np.array(cpu, dtype=float).reshape((1, self.number_of_services))

    def get_service_fail_operational_level(self) -> np.ndarray:
        """get an array of service ftti policies"""
        cpu = []
        for s in self.config:
            if "fail_operational_level" in self.config[s]["policies"]:
                cpu.append(int(self.config[s]["policies"]["fail_operational_level"]))
            else:
                # if not defined fail_operational_level = 0
                cpu.append(0)
        return np.array(cpu, dtype=int).reshape((1, self.number_of_services))

    def get_unicast_adjacency_matrix(self) -> np.matrix:
        """create a unicast adjacency matrix for service communication"""
        A = np.zeros((self.number_of_services, self.number_of_services))

        sender = {}
        for ind, s in enumerate(self.config):
            if self.config[s]["policies"]["network"]["provides"]:
                for ss in self.config[s]["policies"]["network"]["provides"]:
                    if ss["type"] == "unicast":
                        sender[ss["interface"]] = {
                            "index": ind,
                            "bandwidth": ss["size"] / ss["period"],
                        }

        for ind, s in enumerate(self.config):
            if self.config[s]["policies"]["network"]["consumes"]:
                for sr in self.config[s]["policies"]["network"]["consumes"]:
                    if sr["type"] == "unicast":
                        A[sender[sr["interface"]]["index"], ind] = sender[
                            sr["interface"]
                        ]["bandwidth"]

        return np.matrix(A)

    def get_mem_ram(self) -> np.ndarray:
        """get an array of memory policies for each service"""
        mem = []
        for s in self.config:
            mem.append(self.config[s]["policies"]["memory"])
        return np.array(mem, dtype=int)

    def get_active_services(self, scenario: str) -> np.ndarray:
        """
        get an array with all services
        value is 1 if service is active and 0 otherwise
        """
        active_services: np.ndarray = np.ones((self.number_of_services, 1))

        # deactivate services that are in the current scenario
        for s in self.config:
            # if service does not have any scenarios, remains active
            if "scenarios" in self.config[s]:
                # if scenarios are empty, it remains active
                if self.config[s]["scenarios"]:
                    # if scenario is not in the list, it gets deactivated
                    if scenario not in self.config[s]["scenarios"]:
                        ind = self.name_to_id_dict[self.config[s]["name"]]
                        active_services[ind, 0] = 0

        return active_services


class MetricParser:
    """class to create a parser for metric related information"""

    def __init__(self, config: dict):
        self.config: dict = config


class ScenarioParser:
    """class to create a parser for metric related information"""

    def __init__(self, config: dict):
        self.config: dict = config
        self.scenarios: dict[str, list[str]] = {}
        self.has_failure_scenarios: bool = False

        self.__check_for_failure_scenarios()

    def get_operation_scenarios(self) -> dict[str, list[str]]:
        """get the operation scenarios"""

        if "operation_scenarios" in self.config:
            for scenario in self.config["operation_scenarios"]:

                if scenario["name"] not in self.scenarios:
                    self.scenarios[scenario["name"]] = scenario["transitions"]

                    logger.debug(
                        "Found scenario: %s with transitions: %s",
                        scenario["name"],
                        scenario["transitions"],
                    )
        return self.scenarios

    def __check_for_failure_scenarios(self) -> None:
        """check if failure scenarios are activated"""
        if "failure_scenarios" in self.config:
            self.has_failure_scenarios = self.config["failure_scenarios"]


class ConfigParser:
    """class to parse a configuration file and prepare the meta model"""

    def __init__(self) -> None:
        self.config: dict
        self.nodes: NodeParser
        self.services: ServiceParser
        self.metrics: MetricParser
        self.scenarios: ScenarioParser
        self._schema_path: Path | None = Path(META_SCHEMA_PATH) if META_SCHEMA_PATH else None

    def _validate_against_meta_schema(self, config: dict) -> None:
        if self._schema_path is None:
            logger.info(
                "No meta schema env var set (%s); skipping schema validation",
                self._schema_path,
            )
            return
        logger.info(
            "Validating configuration against schemas in: %s", self._schema_path
        )
        validate_with_meta_schema(config, self._schema_path)
        logger.info(
            "Configuration is valid against at least one schema in: %s",
            self._schema_path,
        )

    def _normalize_new_schema_to_old(self, config: dict) -> dict:
        """
        Normalize V2 config (version == 2) into legacy V1 schema (nodes + services + policies).
        For V1 (version == 1 or no version), return a copy unchanged.
        """
        cfg = copy.deepcopy(config)

        version = cfg.get("version", 1)
        if version != 2:
            # V1 or unspecified → already in legacy format
            return cfg

        # -------- V2 → V1: nodes --------
        # V2 nodes use guarantees: { cpu, memory {ram, rom}, network }
        if "nodes" in cfg:
            new_nodes = {}
            for node_id, node_contract in cfg["nodes"].items():
                guarantees = node_contract.get("guarantees", {})
                cpu = guarantees.get("cpu", {})
                mem = guarantees.get("memory", {})
                net = guarantees.get("network", None)

                new_nodes[node_id] = {
                    "name": node_contract.get("name", node_id),
                    "cpu": {
                        "n_cores": cpu.get("n_cores", 0),
                        "frequency": cpu.get("frequency", 0),
                    },
                    # legacy schema only had "memory" as a single value -> map from RAM
                    "memory": mem.get("ram", 0),
                    "network": net if net is not None else 0,
                }

            cfg["nodes"] = new_nodes

        # -------- V2 → V1: services --------
        # V2 uses service_specifications; we convert to legacy "services" + "policies"
        if "service_specifications" in cfg:
            cfg["services"] = {}
            for svc_id, svc_contract in cfg["service_specifications"].items():
                contract = svc_contract.get("contract", {})
                assumptions = contract.get("assumptions", {})
                guarantees = contract.get("guarantees", {})
                requirements = svc_contract.get("requirements", {})

                mem_ass = assumptions.get("memory", {})
                net_ass = assumptions.get("network", {}) or {}
                net_gua = guarantees.get("network", {}) or {}

                policies = {
                    "task_normalized_instructions": assumptions.get(
                        "task_normalized_instructions", 0
                    ),
                    "task_normalized_instructions_init": assumptions.get(
                        "task_normalized_instructions_init", 0
                    ),
                    "task_period": assumptions.get("task_period", 0),
                    "memory": mem_ass.get("ram", 0),
                    "network": {
                        "provides": net_gua.get("provides") or [],
                        "consumes": net_ass.get("consumes") or [],
                    },
                }

                # Map V2 requirements → legacy policy fields
                if "fail_op_level" in requirements:
                    policies["fail_operational_level"] = requirements["fail_op_level"]
                if "recon_time" in requirements:
                    policies["ftti"] = requirements["recon_time"]

                svc_entry = {
                    "name": svc_contract.get("name", svc_id),
                    "image": svc_contract.get("image", ""),
                    "policies": policies,
                }

                scenarios = svc_contract.get("scenarios")
                if scenarios:
                    svc_entry["scenarios"] = list(scenarios)

                cfg["services"][svc_id] = svc_entry

        return cfg

    def parse(self, filepath: str):
        """parse the configuration file"""
        logger.info(
            "Importing the configurations file: %s",
            filepath,
        )
        with open(
            file=filepath,
            mode="r",
            encoding="utf-8",
        ) as stream:
            try:
                raw_config = yaml.safe_load(stream)
                logger.debug(raw_config)
            except yaml.YAMLError as error:
                logger.exception(error)
                raise

        # Validate raw_config against JSON Schema
        self._validate_against_meta_schema(raw_config)

        logger.info(
            "Imported the configurations file: %s",
            filepath,
        )

        # Normalize possible new schema into old internal schema
        config = self._normalize_new_schema_to_old(raw_config)

        self.nodes = NodeParser(config["nodes"])
        self.services = ServiceParser(config["services"])
        self.metrics = MetricParser(config.get("metrics", {}))

        if "scenarios" not in config:
            config["scenarios"] = {}

        if "operation_scenarios" not in config["scenarios"]:
            config["scenarios"]["operation_scenarios"] = [
                {"name": "Default", "transitions": []}
            ]

        self.scenarios = ScenarioParser(config["scenarios"])
