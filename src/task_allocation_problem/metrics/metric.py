# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""Base class for all metrics. Any metric should inherit from the defined Metric class."""

from abc import ABC, abstractmethod
import cvxpy as cvx

from utils.configuration_parser import ConfigParser
from utils.common import ConstantParameters


class Metric(ABC):
    """parent class for all metrics"""

    def __init__(
        self,
        config: ConfigParser,
        scenario: str,
        variables: dict[str, cvx.Variable | cvx.Expression],
        constants: ConstantParameters,
        **kwargs,
    ) -> None:
        self.constraints: list[cvx.Constraint] = []
        self.objective: cvx.Expression
        self.is_linear: bool = False

        self.config = config
        self.scenario = scenario
        self.variables = variables
        self.constants = constants

        self._apply()

    @abstractmethod
    def _apply(self) -> None:
        """abstract method to implement metric logic"""

    def get_constraints(self) -> list[cvx.Constraint]:
        """get a list of constraints for the metric"""
        return self.constraints

    def get_objective(self) -> cvx.Expression:
        """get the objective function for the metric"""
        return self.objective
