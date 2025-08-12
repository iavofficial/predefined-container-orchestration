# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#

import numpy as np
import cvxpy as cvx
from utils.logger import logger
from task_allocation_problem.metrics.metric import Metric


class SchedulerEDF(Metric):
    """
    METRIC CPU SCHEDULER - EDF

    Based on https://docs.kernel.org/scheduler/sched-deadline.html section 3.3

    Constaint: sum(WCET_i / P_i) <= M - (M - 1) · U_max --> schedulability
    Objective: min(sum(WCET_i / P_i) / sum(M)) --> minimize number of hot-standby instances

    ToDo:
    Objective now has the side effect that services are more likely to get
    scheduled on nodes with higher cpu frequency because the relative utilization
    is smaller.
    Alternative approaches:
       - minimize the sum of normalized instructions
       - minimize deviation of average utilization
    """

    def _apply(self):
        self.is_linear = True

        CPU_NORM_INSTR_SERVICES = cvx.Constant(
            self.config.services.get_cpu_service_normalized_instructions()
        )
        P = cvx.Constant(
            np.reciprocal(self.config.services.get_cpu_service_period())
        )  # use reciprocal to avoid division later on

        CPU_FREQ_NODES = cvx.Constant(
            np.diag(np.reciprocal(self.config.nodes.get_cpu_frequencies()))
        )  # use reciprocal to avoid division later on
        M = cvx.Constant(self.config.nodes.get_cpu_cores())

        CPU_NORM_INSTR_SERVICES_LIST = [
            CPU_NORM_INSTR_SERVICES for _ in range(self.constants.n_nodes)
        ]
        CPU_NORM_INSTR_SERVICES_PER_NODE = cvx.vstack(CPU_NORM_INSTR_SERVICES_LIST)

        P_LIST = [P for _ in range(self.constants.n_nodes)]
        P_PER_NODE = cvx.vstack(P_LIST)

        # CPU_FREQ_NODES is in Hz, so WCET will be in seconds --> multiply by 1000 for ms
        WCET = (
            cvx.multiply(self.variables[f"X_{self.scenario}"], CPU_NORM_INSTR_SERVICES_PER_NODE).T
            @ CPU_FREQ_NODES
        ).T * 1000
        U_services = cvx.multiply(WCET, P_PER_NODE)
        U_sum_node = cvx.sum(U_services, axis=1)
        U_max = cvx.max(U_services, axis=1)  # maximum utilization per node

        self.constraints.append(U_sum_node <= M - cvx.multiply(cvx.abs((M - 1)), U_max))

        # minimize total utilization -> minimize number of hot-standby instances.
        self.objective = cvx.sum(U_sum_node) / cvx.sum(M)
