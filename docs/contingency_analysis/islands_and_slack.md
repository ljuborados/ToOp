# Islands, own load flows and the slack per contingency

OpenLoadFlow's security analysis solves the main [synchronous component](../glossary.md#synchronous-component-island)
only, with one slack for all contingencies. `run_contingency_analysis_powsybl` has optional arguments for grids where
that is not enough. They are all off by default, and they need `method="ac"`.

## What the arguments do

| Argument | Effect |
| -------- | ------ |
| `fill_non_main_components` | Reports the other synchronous components too. A contingency outside them gets their base case values, a contingency inside them is solved with its own load flow. |
| `loadflow_contingency_ids` | Solves the given contingencies with their own AC load flow over every island. |
| `slack_overrides` | Gives a contingency another slack, for a contingency that takes the slack generator out. |
| `fill_no_impact_contingencies` | Gives a contingency powsybl leaves out the base case values and status. |
| `monitored_voltage_levels` | Limits the voltage levels whose buses are reported in every contingency. |
| `contingency_voltage_levels` | Reports further voltage levels in single contingencies. |

## Other synchronous components

With `fill_non_main_components` the security analysis runs with every component solved in the base case, whatever the
component mode of the given parameters is. The result then has, for every contingency:

- the rows of the main component from the security analysis,
- the base case rows of the other components, where the contingency has no element in them,
- the rows of a load flow of its own, where the contingency has an element in another component. Branches,
  transformers, injections and buses count by their component, a switch by every component of its voltage level, an
  HVDC line by its converter stations and a tie line by its dangling lines.

A component the base case load flow does not solve, e.g. one without a generator, is not reported.

## Contingencies solved with their own load flow

A contingency in `loadflow_contingency_ids`, or one routed there by `fill_non_main_components`, is applied on a variant
of the network and solved with one AC load flow over every island. Its rows have the format of the security analysis,
so they go through the same result tables.

The outage is applied like this:

- A switch is opened.
- A branch, transformer or injection is disconnected by the circuit breakers powsybl opens for it. Where powsybl finds
  none to open, e.g. behind a fictitious breaker or with disconnectors only, every closed switch at the nodes of the
  element is opened.
- An HVDC line goes out with both converter stations, a tie line with both dangling lines.
- A bus breaker bus loses its injections, and its branches and transformers on the side that is on the bus.
- A busbar section goes out with the circuit breakers around it. This does not depend on `contingencyPropagation`: an
  N-1 definition run with the propagation switched off has to bring its busbar contingencies propagated already, see
  [contingency propagation](propagation.md#propagation-of-the-dc-solver).

The status of the contingency is the worst status of the islands it can affect: the main component and the islands
reaching into a voltage level of an outaged element. An island the contingency does not touch that fails to solve says
nothing about the contingency. With a slack override, the islands holding one of the given slacks are judged and
reported.

The [connectivity result](limit_table.md#connectivity) lists the elements of the contingency and the branches,
transformers and injections that lost their bus with them. The outage group id is the contingency id, as for the
contingencies the security analysis solves.

## Slack per contingency

`slack_overrides` maps a contingency id to the slack to use, as injection ids, bus breaker bus ids or bus ids. An
injection stands for the bus it is connected to after the outage, so the slack follows a generator when the outage
renumbers the buses. Ids that are not connected or not known are left out with a warning.

Contingencies with the same override share one security analysis. A contingency that is also solved with its own load
flow takes the override there.

## Contingencies without a result of powsybl

Powsybl leaves out a contingency that does not change the network, e.g. the outage of an element that is already
disconnected. By default such a contingency has no result rows and the status `NO_CALCULATION`, as described under
[contingency propagation](propagation.md#powsybl-contingency-propagation). With `fill_no_impact_contingencies` it gets
the base case rows and the status of the base case.

A contingency without elements other than the base case is never passed to powsybl. It always gets the base case
values, the voltage angle differences of the monitored switches included.

## Fewer bus results

Bus results are the largest part of a security analysis result. `monitored_voltage_levels` replaces the voltage levels
the translation of the N-1 definition picks: a monitored bus outside them has no result.
`contingency_voltage_levels` adds voltage levels for single contingencies. They are reported in those contingencies
and in the base case, so the deviation of a voltage from its base case value is there.

## Voltage angle difference across a switch the contingency opens

A monitored switch that is closed in the base case and opened by a contingency gets a voltage angle difference in that
contingency. Each side is read at the busbar section it is still connected to without the switch, see
`get_busbar_bus_breaker_buses`. A side that reaches no busbar section, e.g. a feeder behind its only breaker, has no
angle and the difference is empty. So is the difference between two islands, which have no common angle reference.
