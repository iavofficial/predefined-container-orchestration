# Examples

## Scheduler - EDF
Example to use only the SchedulerEDF metric. This will disable all other metrics.

Configuration:
* 2 single-core nodes
* 2 services with more than 50% cpu utilization on the nodes

TAP will allocate both services on a different node because otherwise the EDF schedulability constraint would be violated.

```bash
make run-example FILENAME=scheduler_edf_config.yml
```

## Memory - RAM
Example to use only the MemoryRAM metric. This will disable all other metrics.

Configuration:
* 2 nodes. One with 250MB RAM and one with 1000MB RAM
* 2 services which require both 300MB RAM

TAP will allocate both services on the node with 1000MB RAM because otherwise the MemoryRAM constraint would be violated.

```bash
make run-example FILENAME=memory_ram_config.yml
```

## Network Load
Example to use only the NetworkLoad metric. This will disable all other metrics.

Configuration:
* 2 nodes
* 2 services that depend on each other

TAP will allocate both services on the same node because this minimizes the objective value (the used network bandwidth / network load)

```bash
make run-example FILENAME=network_load_config.yml
```

## Failure Scenarios
Example to use only the OperationFailureScenario metric. This will disable all other metrics. The current objective function is quadratic. Therefore, solver SCIP will be used.

Configuration:
* 3 nodes
* 3 services with no dependencies

TAP will allocate all services on different nodes because this minimizes the objective value (the number of reconfigurations in case of a failure)

```bash
make run-example FILENAME=failure_scenarios_config.yml
```

## All Metrics
Example how to use all metrics. Currently, it is just some arbitrary data. It would be better to design it in a way, so that it can be easily checked if the algorithm works as expected (e.g. by creating some service clusters and checking that they are assigned on the same nodes).

```bash
make run-example FILENAME=all_metrics_config.yml
```
