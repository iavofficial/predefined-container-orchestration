# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""this module provides network related metrics"""

import numpy as np
import cvxpy as cvx
from utils.logger import logger
from task_allocation_problem.metrics.metric import Metric


class NetworkLoadQuad(Metric):
    """
    METRIC NETWORK LOAD

    Calculates the unicast network load from service dependencies
    ToDo: Add multicast network load

    Constraint:
        - node_port_load_egress <= max_node_port_load
        - node_port_load_ingress <= max_node_port_load
    Objective: minimize network load

    CANNOT BE USED:
    X @ adj_services @ X.T does not follow DCP rules of cvxpy
    for x @ Q @ x.T cvx.quad_form could be used, but we have a matrix
    and not a vector.
    Maybe it would be possible to split X in vectors, perform quad_form
    for each vector und then put it back together using vstack / hstack
    """

    def _apply(self):
        self.is_linear = False
        adj_services = self.config.services.get_unicast_adjacency_matrix()
        max_port_load_node = self.config.nodes.get_net_bandwidth()

        H = cvx.Constant(
            np.ones(shape=(self.constants.n_nodes, self.constants.n_nodes))
            - np.identity(self.constants.n_nodes)
        )
        network_load = cvx.multiply(
            self.variables[f"X_{self.scenario}"] @ adj_services @ self.variables[f"X_{self.scenario}"].T, H
        )

        NN = cvx.Constant(max_port_load_node)
        NN_total_rec = cvx.Constant(1 / np.sum(max_port_load_node))

        # max net utilization constraints
        self.constraints.append(cvx.sum(network_load, axis=0) <= NN)  # egress
        self.constraints.append(cvx.sum(network_load, axis=1) <= NN)  # ingress

        # normalize as sum(load)/sum(bandwidth) --> sum(bandwidth) = max
        obj_net_unicast = cvx.sum(network_load) * NN_total_rec
        obj_net_multicast = ()
        self.objective = obj_net_unicast


class NetworkLoad(Metric):
    """
    METRIC NETWORK LOAD

    Calculates the unicast network load from service dependencies
    ToDo: Add multicast network load

    Constraint:
        - node_port_load_egress <= max_node_port_load
        - node_port_load_ingress <= max_node_port_load
    Objective: minimize network load
    """

    def _apply(self):
        self.is_linear = True
        adj_services = self.config.services.get_unicast_adjacency_matrix()
        max_port_load_node = self.config.nodes.get_net_bandwidth()

        AA = cvx.Constant(
            self.__create_block_matrix(
                adj_services, adj_services, self.constants.n_nodes, self.constants.n_services
            )
        )
        logger.debug(AA.value)
        JO = cvx.Constant(
            self.__create_block_matrix(
                np.ones((1, self.constants.n_services)),
                np.zeros((1, self.constants.n_services)),
                self.constants.n_nodes,
                self.constants.n_services,
            )
        )
        logger.debug(JO.value)
        OJ = cvx.Constant(
            np.ones((self.constants.n_nodes, self.constants.n_nodes))
            - np.identity(self.constants.n_nodes)
        )
        logger.debug(OJ.value)

        NN = cvx.Constant(max_port_load_node)
        NN_rec = cvx.Constant(np.reciprocal(NN.value))

        # Linearize X @ adj_services @ X.T
        # https://yetanothermathprogrammingconsultant.blogspot.com/2019/11/cvxpy-matrix-style-modeling-limits.html
        # Z <= x*e^T
        # Z <= e*x^T
        # Z >= x*e^T + e*x^T-e*e^T
        #
        # https://or.stackexchange.com/questions/37/how-to-linearize-the-product-of-two-binary-variables

        Y = cvx.Variable(
            (
                self.constants.n_nodes * self.constants.n_services,
                self.constants.n_nodes * self.constants.n_services,
            ),
            boolean=True,
        )

        for i in range(self.constants.n_nodes * self.constants.n_services):
            for j in range(self.constants.n_nodes * self.constants.n_services):
                self.constraints.append(
                    Y[i][j] <= cvx.vec(self.variables[f"X_{self.scenario}"], order="C")[i]
                )
                self.constraints.append(
                    Y[i][j] <= cvx.vec(self.variables[f"X_{self.scenario}"], order="C")[j]
                )
                self.constraints.append(
                    Y[i][j]
                    >= cvx.vec(self.variables[f"X_{self.scenario}"], order="C")[i]
                    + cvx.vec(self.variables[f"X_{self.scenario}"], order="C")[j]
                    - 1
                )

        Z = cvx.multiply(
            JO @ cvx.multiply(AA, Y) @ JO.T, OJ
        )  # Workaorund for Frobenius inner product of block matrices

        # max net utilization constraints
        self.constraints.append(cvx.sum(Z, axis=0) <= NN)  # egress
        self.constraints.append(cvx.sum(Z, axis=1) <= NN)  # ingress

        obj_net_unicast = (
            self.constants.ones_vector_n_nodes @ Z @ NN_rec.T
            + NN_rec @ (Z @ self.constants.ones_vector_n_nodes.T)
        ) / (2 * self.constants.n_nodes)
        obj_net_multicast = ()
        self.objective = obj_net_unicast

    def __create_block_matrix(
        self, Diag: np.ndarray, Non_Diag: np.ndarray, n_rows: int, n_cols: int
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
                    else:
                        row = np.concatenate((row, Non_Diag), axis=1)

            if i == 0:
                B = row
            else:
                B = np.concatenate((B, row), axis=0)

        return B
