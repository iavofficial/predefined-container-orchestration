# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""this module provides memory related metrics"""
import sys
import cvxpy as cvx
from utils.logger import logger
from task_allocation_problem.metrics.metric import Metric


class MemoryRAM(Metric):
    """
    METRIC MEMORY - RAM

    Constaint:
        sum of required memory from assigned services
        does not exceed provided memory from node
    Objective: -
    """

    def _apply(self):

        if self.config.metrics.config[self.__class__.__name__]["weight"] != 0:
            logger.error(
                "%s weight must be 0, because no objective is implemented",
                str(self.__class__.__name__),
            )
            sys.exit(1)
        self.is_linear = True

        MEM_RAM_SERVICES = cvx.Constant(self.config.services.get_mem_ram())
        MEM_RAM_NODES = cvx.Constant(self.config.nodes.get_mem_ram())

        self.constraints.append(
            self.variables[f"X_{self.scenario}"] @ MEM_RAM_SERVICES <= MEM_RAM_NODES
        )
        self.objective = cvx.Constant(0)

class MemoryROM(Metric):
    """
    METRIC MEMORY - ROM

    Constaint:
        sum of required memory from assigned services
        does not exceed provided memory from node
    Objective: -
    """

    def _apply(self):

        if self.config.metrics.config[self.__class__.__name__]["weight"] != 0:
            logger.error(
                "%s weight must be 0, because no objective is implemented",
                str(self.__class__.__name__),
            )
            sys.exit(1)
        self.is_linear = True

        MEM_ROM_SERVICES = cvx.Constant(self.config.services.get_mem_rom())
        MEM_ROM_NODES = cvx.Constant(self.config.nodes.get_mem_rom())

        self.constraints.append(
            self.variables[f"X_{self.scenario}"] @ MEM_ROM_SERVICES <= MEM_ROM_NODES
        )
        self.objective = cvx.Constant(0)
