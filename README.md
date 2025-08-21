<!---
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

  Created by Lukas Stahlbock in 2024
  Copyright (c) 2024 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.

-->
# Task Allocation Problem for predefined container orchestration configuration

Implementation of MILP to obtain orchestrations for a set of services on a set of nodes in a predefined reconfiguration scenario. Different metrics are used to define constraints and objectives for the problem. This leads to a multi-objective optimization problem which is formulated as a single objective problem using a weighted sum approach. Therefore, all metrics should be normalized to a range of 0-1.

Currently included metrics are:

| Name | Order |
| -----| ------|
| SchedulerEDF | linear |
| MemoryRAM | linear |
| NetworkLoad | linear (quadratic but has been linearized) |
| FaultToleranceTimeInterval | linear (quadratic bus has been linearized) |

The algorithm will check the order of all metrics. If all metrics are linear, CBC solver will be used. If any metric is not linear, SCIP solver will be used.

The Task Allocation Problem (TAP) will obtain a matrix where each row resembles a node and each column resembles a service instance. If a service instance is assigned to a node the value in the matrix is 1, otherwise it is 0.

## Setup
* Python 3.12 must be installed
* ```
  pip install uv
  uv venv --python 3.12
  uv pip install -r pyproject.toml
  ```

Or just use the provided Devcontainer (preferrably with VS Code Devcontainer extension `ms-vscode-remote.remote-containers`).

## Steps to add a new metric
1. create a new class in task_allocation_problem/metrics directory
2. add the class in `__init__.py` of task_allocation_problem/metrics directory
3. add the class to a TAP config file

The used framework cvxpy enforces disciplined convex programming (DCP) rules. More information on DCP can be found [here](https://www.cvxpy.org/tutorial/dcp/index.html). Some general tips and tricks for modeling problems can be found [here](https://docs.mosek.com/modeling-cookbook/mio.html)

## Examples

Some example configurations are provided in the `examples` directory and explainations are given in [Examples Subsection](doc/examples.md).

## License

Licensed under the Apache License, Version 2.0 (the "License"); <br>
you may not use this file except in compliance with the License. <br>
You may obtain a copy of the License at <br>
<br>
http://www.apache.org/licenses/LICENSE-2.0
