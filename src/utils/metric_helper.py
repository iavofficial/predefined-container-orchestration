# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""module provides helper functions for metrics"""

import numpy as np


def create_block_matrix(
    Diag: np.ndarray, Non_Diag: np.ndarray, n_rows: int, n_cols: int
) -> np.ndarray:
    """
    create a block matrix with the matrix Diag on the diagonal elements and
    Non_Diag on the non diagonal elements
    """
    B = np.zeros(shape=(n_cols, n_rows * n_cols))
    for i in range(n_rows):
        row = Non_Diag
        for j in range(n_rows):
            if i == j:
                if j == 0:
                    row = Diag
                else:
                    row = np.concatenate((row, Diag), axis=1)
            else:
                if j == 0:
                    continue

                row = np.concatenate((row, Non_Diag), axis=1)

        if i == 0:
            B = row
        else:
            B = np.concatenate((B, row), axis=0)

    return B
