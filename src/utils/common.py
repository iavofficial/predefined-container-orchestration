# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""module provides common datastructures"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ConstantParameters:
    """group constant parameters"""

    n_nodes: int
    n_services: int
    ones_vector_n_nodes: np.ndarray
