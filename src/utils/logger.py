# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""module provides a logger"""
import logging

# Define logger
logger = logging.getLogger(__name__)


def init_logger(log_level):
    """set the formatting for logger"""
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
