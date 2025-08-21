# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2024
#  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#
"""entrypoint to create the TAP"""

import os
import argparse

from utils.logger import logger, init_logger
from task_allocation_problem.task_allocation_problem import TaskAllocationProblem


def main() -> None:
    """the main function"""
    parser = argparse.ArgumentParser(
        description="Offline Algorithm for predefined reconfigurations"
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        required=False,
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="The log level. Default is INFO.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        help="Input config file containing service, node and TAP configurations",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=False,
        type=str,
        help="Directory path for generated output",
    )

    args = parser.parse_args()
    init_logger(args.log_level)
    logger.debug("Initialized logger")
    verbose: bool = False
    if args.log_level == "DEBUG":
        verbose = True

    output_dir: str = os.path.join(os.getcwd(), "output")
    if args.output_dir:
        output_dir = args.output_dir

    tap = TaskAllocationProblem()
    tap.import_configurations(args.config)
    tap.define_objectives_and_constraints()
    tap.solve(verbose=verbose)
    tap.print_solution()
    tap.generate(output_dir)


if __name__ == "__main__":
    main()
