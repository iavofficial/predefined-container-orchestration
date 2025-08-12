# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""Module to import metrics into the TAP"""

from enum import Enum

from task_allocation_problem.metrics.metric import Metric
from task_allocation_problem.metrics.network import NetworkLoad, NetworkLoadQuad
from task_allocation_problem.metrics.scheduler import SchedulerEDF
from task_allocation_problem.metrics.memory import MemoryRAM
from task_allocation_problem.metrics.scenarios import FaultToleranceTimeInterval

__all__ = ["get_metric"]


METRICS: dict[str, type[Metric]] = {
    "SchedulerEDF": SchedulerEDF,
    "MemoryRAM": MemoryRAM,
    "NetworkLoad": NetworkLoad,
}

SCENARIO_METRICS: dict[str, type[Metric]] = {
    "FaultToleranceTimeInterval": FaultToleranceTimeInterval
}


def get_metric(name: str) -> type[Metric]:
    """
    Dynamically import a Metric
    """

    return METRICS[name]


def get_scenario_metric(name: str) -> type[Metric]:
    """
    Dynamically import a Metric
    """

    return SCENARIO_METRICS[name]
