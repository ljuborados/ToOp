# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Raw security analysis results for what OpenLoadFlow's security analysis does not compute itself.

OpenLoadFlow solves only the main synchronous component after a contingency, and it leaves out contingencies that do
not change the network. The helpers here fill those gaps on the raw result frames (the frames of
``get_ca_bus_results`` and friends), so the rest of the result processing does not need to know where a row came from:

- contingencies without impact get the base case rows,
- untouched synchronous components other than the main one get their base case rows,
- contingencies that need their own load flow (every island, or a given slack) are solved one by one on a variant of
  the network, and their results are written in the raw security analysis format.
"""

import uuid
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd
import polars as pl
import pypowsybl
import structlog
from pypowsybl.loadflow import ComponentMode, ComponentStatus
from pypowsybl.network import Network

logger = structlog.get_logger(__name__)

# The contingency id powsybl gives the pre-contingency rows of a security analysis
BASECASE_IN_RESULT = ""

BUS_RESULT_SCHEMA = {
    "contingency_id": pl.String,
    "operator_strategy_id": pl.String,
    "voltage_level_id": pl.String,
    "bus_id": pl.String,
    "v_mag": pl.Float64,
    "v_angle": pl.Float64,
}
BRANCH_RESULT_SCHEMA = {
    "contingency_id": pl.String,
    "operator_strategy_id": pl.String,
    "branch_id": pl.String,
    "p1": pl.Float64,
    "q1": pl.Float64,
    "i1": pl.Float64,
    "p2": pl.Float64,
    "q2": pl.Float64,
    "i2": pl.Float64,
    "flow_transfer": pl.Float64,
}
BUS_COMPONENT_SCHEMA = {
    "contingency_id": pl.String,
    "bus_id": pl.String,
    "synchronous_component": pl.Int64,
}
THREE_WINDINGS_RESULT_SCHEMA = {
    "contingency_id": pl.String,
    "transformer_id": pl.String,
    "p1": pl.Float64,
    "q1": pl.Float64,
    "i1": pl.Float64,
    "p2": pl.Float64,
    "q2": pl.Float64,
    "i2": pl.Float64,
    "p3": pl.Float64,
    "q3": pl.Float64,
    "i3": pl.Float64,
}


@dataclass
class RawSecurityResults:
    """The raw result frames of a security analysis: one row per contingency and bus, branch or 3w transformer."""

    bus_results: pl.LazyFrame
    branch_results: pl.LazyFrame
    three_windings_transformer_results: pl.LazyFrame
    bus_components: pl.LazyFrame | None = None
    """The synchronous component of the bus rows, where the rows of a contingency span several islands.

    Islands are solved with a slack each, so angles of different islands have no common reference. Without this frame
    all bus rows of a contingency are taken to be in one island, which is what a security analysis reports.
    """

    @classmethod
    def empty(cls) -> "RawSecurityResults":
        """Frames without rows, with the raw security analysis columns."""
        return cls(
            pl.LazyFrame(schema=BUS_RESULT_SCHEMA),
            pl.LazyFrame(schema=BRANCH_RESULT_SCHEMA),
            pl.LazyFrame(schema=THREE_WINDINGS_RESULT_SCHEMA),
        )


@dataclass
class SynchronousComponents:
    """The synchronous component of every bus result id, branch and three winding transformer of a network.

    Component 0 is the main one: powsybl numbers components by size and OpenLoadFlow's security analysis solves only
    component 0. Disconnected elements are not in the maps.
    """

    bus: dict[str, int]
    branch: dict[str, int]
    three_windings_transformer: dict[str, int]
    voltage_level: dict[str, set[int]]
    injection: dict[str, int]
    solved: set[int] | None = None
    """The components the base case load flow solves; None counts every component as solved."""

    def is_non_main(self, component: int) -> bool:
        """Whether *component* is solved in the base case but is not the main component."""
        return component != 0 and (self.solved is None or component in self.solved)


def _conform(frame: pl.LazyFrame, schema: Mapping[str, pl.DataType]) -> pl.LazyFrame:
    names = frame.collect_schema().names()
    return frame.select(
        [
            pl.col(name).cast(dtype) if name in names else pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in schema.items()
        ]
    )


def conform_raw_results(raw: RawSecurityResults) -> RawSecurityResults:
    """Give the raw frames exactly the canonical columns and dtypes.

    An empty powsybl result comes with null-typed id columns, which cannot be concatenated with filled ones.
    """
    return RawSecurityResults(
        _conform(raw.bus_results, BUS_RESULT_SCHEMA),
        _conform(raw.branch_results, BRANCH_RESULT_SCHEMA),
        _conform(raw.three_windings_transformer_results, THREE_WINDINGS_RESULT_SCHEMA),
        _conform(raw.bus_components, BUS_COMPONENT_SCHEMA) if raw.bus_components is not None else None,
    )


def concat_raw_results(results: Sequence[RawSecurityResults]) -> RawSecurityResults:
    """Concatenate the raw results of several runs."""
    conformed = [conform_raw_results(result) for result in results]
    if not conformed:
        return RawSecurityResults.empty()
    bus_components = [result.bus_components for result in conformed if result.bus_components is not None]
    return RawSecurityResults(
        pl.concat([result.bus_results for result in conformed]),
        pl.concat([result.branch_results for result in conformed]),
        pl.concat([result.three_windings_transformer_results for result in conformed]),
        pl.concat(bus_components) if bus_components else None,
    )


def _relabel_basecase_rows(
    frame: pl.LazyFrame, id_column: str, contingency_ids: Collection[str], keep_ids: Collection[str] | None
) -> pl.LazyFrame:
    columns = frame.collect_schema().names()
    base = frame.filter(pl.col("contingency_id") == BASECASE_IN_RESULT)
    if keep_ids is not None:
        base = base.filter(pl.col(id_column).is_in(list(keep_ids)))
    targets = pl.LazyFrame({"contingency_id": list(contingency_ids)}, schema={"contingency_id": pl.String})
    copies = base.drop("contingency_id").join(targets, how="cross").select(columns)
    # The analysis does report a component other than the main one where the contingency changes it (an HVDC link
    # into it, say); those rows stand
    return copies.join(frame.select("contingency_id", id_column), on=["contingency_id", id_column], how="anti")


def copy_basecase_rows(
    raw: RawSecurityResults,
    contingency_ids: Collection[str],
    components: SynchronousComponents | None = None,
) -> RawSecurityResults:
    """Append the base case rows once more for each of *contingency_ids*.

    Parameters
    ----------
    raw : RawSecurityResults
        Conformed raw results holding the base case rows (contingency id ``""``).
    contingency_ids : Collection[str]
        The contingencies that get the copies.
    components : SynchronousComponents | None
        If given, only the rows of buses and branches outside the main synchronous component are copied. Without it all
        base case rows are copied, which is what a contingency without impact needs.

    Returns
    -------
    RawSecurityResults
        *raw* with the copies appended.
    """
    if not contingency_ids:
        return raw

    def non_main(mapping: Mapping[str, int]) -> list[str] | None:
        if components is None:
            return None
        return [element for element, component in mapping.items() if components.is_non_main(component)]

    copies = RawSecurityResults(
        _relabel_basecase_rows(raw.bus_results, "bus_id", contingency_ids, non_main(components.bus if components else {})),
        _relabel_basecase_rows(
            raw.branch_results, "branch_id", contingency_ids, non_main(components.branch if components else {})
        ),
        _relabel_basecase_rows(
            raw.three_windings_transformer_results,
            "transformer_id",
            contingency_ids,
            non_main(components.three_windings_transformer if components else {}),
        ),
    )
    return concat_raw_results([raw, copies])


def _solved_components(net: Network, lf_params: pypowsybl.loadflow.Parameters) -> set[int]:
    """Return the synchronous components an all-component base case load flow solves (on a scratch variant)."""
    base_variant = net.get_working_variant_id()
    variant = f"components-{uuid.uuid4()}"
    net.clone_variant(base_variant, variant)
    net.set_working_variant(variant)
    try:
        pypowsybl.loadflow.run_ac(net, with_slack_buses(lf_params, None, all_components=True))
        buses = net.get_buses(attributes=["synchronous_component", "v_mag"])
        return set(buses.loc[buses["v_mag"].notna(), "synchronous_component"].astype(int))
    finally:
        net.set_working_variant(base_variant)
        net.remove_variant(variant)


def get_synchronous_components(
    net: Network, lf_params: pypowsybl.loadflow.Parameters | None = None
) -> SynchronousComponents:
    """Map buses, branches, transformers, injections and voltage levels to their synchronous components.

    Bus result ids are busbar sections and bus breaker view buses, the ids a security analysis reports buses by. With
    *lf_params*, the components a base case load flow does not solve (dead islands) do not count as components to
    report; without, every component does.
    """
    buses = net.get_buses(attributes=["synchronous_component", "voltage_level_id"])
    component_of_bus = buses["synchronous_component"]

    def map_bus(bus_ids: pd.Series) -> pd.Series:
        return bus_ids.map(component_of_bus)

    def to_dict(series: pd.Series) -> dict[str, int]:
        series = series.dropna()
        return {str(key): int(value) for key, value in series.items()}

    busbar_sections = net.get_busbar_sections(attributes=["bus_id"])
    bus_breaker_buses = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    bus = {**to_dict(map_bus(bus_breaker_buses["bus_id"])), **to_dict(map_bus(busbar_sections["bus_id"]))}

    branches = net.get_branches(attributes=["bus1_id", "bus2_id"])
    branch = to_dict(map_bus(branches["bus1_id"]).fillna(map_bus(branches["bus2_id"])))
    transformers = net.get_3_windings_transformers(attributes=["bus1_id", "bus2_id", "bus3_id"])
    three_windings = to_dict(
        map_bus(transformers["bus1_id"]).fillna(map_bus(transformers["bus2_id"])).fillna(map_bus(transformers["bus3_id"]))
    )
    injections = net.get_injections(attributes=["bus_id"])
    injection = to_dict(map_bus(injections["bus_id"]))
    voltage_level: dict[str, set[int]] = {}
    for voltage_level_id, component in zip(buses["voltage_level_id"], component_of_bus, strict=True):
        voltage_level.setdefault(str(voltage_level_id), set()).add(int(component))
    solved = _solved_components(net, lf_params) if lf_params is not None else None
    return SynchronousComponents(bus, branch, three_windings, voltage_level, injection, solved)


def contingencies_touching_non_main_components(
    net: Network, contingencies: Mapping[str, Sequence[str]], components: SynchronousComponents
) -> set[str]:
    """Contingencies with an element in a synchronous component other than the main one.

    The security analysis cannot solve those components, so these contingencies need their own load flow. A switch
    counts as touching every component of its voltage level, an HVDC line the components of its converter stations
    and a tie line those of its dangling lines.
    """
    switch_voltage_levels = net.get_switches(attributes=["voltage_level_id"])["voltage_level_id"].to_dict()
    wanted = {element for elements in contingencies.values() for element in elements}
    parts = _element_parts(net, wanted)
    touching = set()
    for contingency_id, elements in contingencies.items():
        for element in elements:
            element_components: set[int] = set()
            for part in parts.get(element, [element]):
                for mapping in (
                    components.branch,
                    components.three_windings_transformer,
                    components.injection,
                    components.bus,
                ):
                    if part in mapping:
                        element_components.add(mapping[part])
            if element in switch_voltage_levels:
                element_components |= components.voltage_level.get(str(switch_voltage_levels[element]), set())
            if any(components.is_non_main(component) for component in element_components):
                touching.add(contingency_id)
                break
    return touching


_STATUS_SEVERITY = {
    ComponentStatus.FAILED: 3,
    ComponentStatus.MAX_ITERATION_REACHED: 2,
    ComponentStatus.CONVERGED: 1,
    ComponentStatus.NO_CALCULATION: 0,
}


def _overall_status(component_results: Sequence, components: Collection[int] | None = None) -> ComponentStatus:
    """Return the worst status of the solved components, as one load flow over the whole network would.

    With *components*, only those synchronous components count.
    """
    statuses = [
        result.status for result in component_results if components is None or result.synchronous_component_num in components
    ] or [ComponentStatus.NO_CALCULATION]
    return max(statuses, key=lambda status: _STATUS_SEVERITY.get(status, 3))


@dataclass
class _OutageElements:
    """What applying a contingency needs to know of its elements, read once from the base network."""

    types: pd.Series
    """Identifiable id -> type."""
    parts: dict[str, list[str]]
    """HVDC line, tie line or bus breaker bus id -> the connectables that are disconnected for it."""
    voltage_levels: dict[str, set[str]]
    """Contingency element id -> its voltage levels."""
    node_breaker_levels: set[str]
    """The voltage levels in node-breaker topology."""
    bus_terminals: dict[str, list[tuple[str, int | None]]] = field(default_factory=dict)
    """Bus breaker bus id -> what is connected to it, see ``_bus_terminals``."""


def _element_parts(net: Network, element_ids: Collection[str]) -> dict[str, list[str]]:
    """Return the connectables that stand for the elements that are no connectable themselves.

    An HVDC line goes out with both converter stations and a tie line with both dangling lines.
    """
    wanted = list(element_ids)
    parts: dict[str, list[str]] = {}
    hvdc_lines = net.get_hvdc_lines(attributes=["converter_station1_id", "converter_station2_id"])
    tie_lines = net.get_tie_lines(attributes=["dangling_line1_id", "dangling_line2_id"])
    for frame in (hvdc_lines, tie_lines):
        for element in frame.index.intersection(wanted):
            parts[str(element)] = [str(part) for part in frame.loc[element]]
    return parts


def _bus_terminals(net: Network, element_ids: Collection[str]) -> dict[str, list[tuple[str, int | None]]]:
    """Return what is connected to the bus breaker buses among *element_ids*: (connectable, side).

    The side is None for an injection. Busbar sections are left out.
    """
    buses = net.get_bus_breaker_view_buses(attributes=[]).index.intersection(list(element_ids))
    if not len(buses):
        return {}
    terminals: dict[str, list[tuple[str, int | None]]] = {str(bus): [] for bus in buses}
    injections = net.get_injections(attributes=["type", "bus_breaker_bus_id"]).query("type != 'BUSBAR_SECTION'")
    branches = net.get_branches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"])
    transformers = net.get_3_windings_transformers(
        attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "bus_breaker_bus3_id"]
    )
    columns = [(injections, None, "bus_breaker_bus_id")]
    columns += [(branches, side, f"bus_breaker_bus{side}_id") for side in (1, 2)]
    columns += [(transformers, side, f"bus_breaker_bus{side}_id") for side in (1, 2, 3)]
    for frame, side, column in columns:
        connected = frame[frame[column].isin(terminals)]
        for connectable, bus in zip(connected.index, connected[column], strict=True):
            terminals[str(bus)].append((str(connectable), side))
    return terminals


def _outage_elements(net: Network, element_ids: Collection[str]) -> _OutageElements:
    types = net.get_identifiables(attributes=["type"])["type"]
    parts = _element_parts(net, element_ids)
    bus_terminals = _bus_terminals(net, element_ids)

    wanted = {*element_ids, *(part for element_parts in parts.values() for part in element_parts)}
    wanted |= {connectable for terminals in bus_terminals.values() for connectable, _ in terminals}
    voltage_levels: dict[str, set[str]] = {}
    frames = (
        net.get_branches(attributes=["voltage_level1_id", "voltage_level2_id"]),
        net.get_3_windings_transformers(attributes=["voltage_level1_id", "voltage_level2_id", "voltage_level3_id"]),
        net.get_injections(attributes=["voltage_level_id"]),
        net.get_busbar_sections(attributes=["voltage_level_id"]),
        net.get_switches(attributes=["voltage_level_id"]),
        net.get_bus_breaker_view_buses(attributes=["voltage_level_id"]),
    )
    for frame in frames:
        rows = frame[frame.index.isin(list(wanted))]
        for element, row in zip(rows.index, rows.to_numpy(), strict=True):
            voltage_levels.setdefault(str(element), set()).update(str(level) for level in row)
    for element, element_parts in parts.items():
        voltage_levels[element] = voltage_levels.get(element, set()).union(
            *(voltage_levels.get(part, set()) for part in element_parts)
        )
    topology_kinds = net.get_voltage_levels(attributes=["topology_kind"])["topology_kind"]
    node_breaker_levels = set(topology_kinds.index[topology_kinds == "NODE_BREAKER"].astype(str))
    return _OutageElements(types, parts, voltage_levels, node_breaker_levels, bus_terminals)


def _has_bus(net: Network, element: str, element_type: str | None) -> bool:
    """Whether a side of *element* is still on a bus of the working variant."""
    if element_type == "THREE_WINDINGS_TRANSFORMER":
        frame = net.get_3_windings_transformers(id=element, attributes=["bus1_id", "bus2_id", "bus3_id"])
    elif element_type == "BUSBAR_SECTION":
        frame = net.get_busbar_sections(id=element, attributes=["bus_id"])
    elif element_type in ("LINE", "TWO_WINDINGS_TRANSFORMER"):
        frame = net.get_branches(id=element, attributes=["bus1_id", "bus2_id"])
    else:
        frame = net.get_injections(id=element, attributes=["bus_id"])
    return bool((frame.fillna("") != "").to_numpy().any())


def _open_switches_at(net: Network, element: str, outage_elements: _OutageElements) -> bool:
    """Open every closed switch at the nodes of *element*, whatever its kind; False if it has no such node.

    This takes the element out where powsybl finds no circuit breaker to open for it: behind a fictitious breaker,
    or connected through disconnectors alone.
    """
    to_open: list[str] = []
    found = False
    for voltage_level in outage_elements.voltage_levels.get(element, set()) & outage_elements.node_breaker_levels:
        topology = net.get_node_breaker_topology(voltage_level)
        nodes = set(topology.nodes.index[topology.nodes["connectable_id"] == element].astype(int))
        if not nodes:
            continue
        found = True
        # Nodes joined by an internal connection are one node
        connections = topology.internal_connections[["node1", "node2"]].to_numpy()
        grown = True
        while grown:
            joined = {int(node) for pair in connections if set(pair) & nodes for node in pair}
            grown = not joined <= nodes
            nodes |= joined
        switches = topology.switches
        at_nodes = (switches["node1"].isin(nodes) | switches["node2"].isin(nodes)) & ~switches["open"]
        to_open += switches.index[at_nodes].astype(str).to_list()
    if to_open:
        net.update_switches(pd.DataFrame({"open": True}, index=pd.Index(to_open, name="id")))
    return found


def _disconnect(net: Network, element: str, outage_elements: _OutageElements) -> bool:
    """Disconnect a connectable; an element that is off already counts as disconnected."""
    element_type = outage_elements.types.get(element)
    try:
        if net.disconnect(element):
            return True
        # powsybl returns False as well for an element that is dead already, for instance one the breakers of the
        # same contingency have just isolated
        if not _has_bus(net, element, element_type):
            return True
        return _open_switches_at(net, element, outage_elements) and not _has_bus(net, element, element_type)
    except pypowsybl.PyPowsyblError as error:
        logger.warning("powsybl_outage_not_applied", element=element, error=str(error))
        return False


_SIDE_UPDATES = {
    "LINE": "update_lines",
    "TWO_WINDINGS_TRANSFORMER": "update_2_windings_transformers",
    "THREE_WINDINGS_TRANSFORMER": "update_3_windings_transformers",
}


def _disconnect_from_bus(net: Network, bus_element: str, outage_elements: _OutageElements) -> bool:
    """Take everything off a bus breaker bus, as the security analysis loses a bus.

    Injections go out, branches and transformers only on the side that is on the bus.
    """
    applied = True
    for connectable, side in outage_elements.bus_terminals.get(bus_element, []):
        update = _SIDE_UPDATES.get(outage_elements.types.get(connectable))
        if side is None or update is None:
            applied &= _disconnect(net, connectable, outage_elements)
            continue
        try:
            getattr(net, update)(pd.DataFrame({f"connected{side}": False}, index=pd.Index([connectable], name="id")))
        except pypowsybl.PyPowsyblError as error:
            logger.warning("powsybl_outage_not_applied", element=connectable, error=str(error))
            applied = False
    return applied


def _apply_outage(net: Network, elements: Sequence[str], outage_elements: _OutageElements) -> bool:
    """Open the switches and disconnect the other elements of a contingency on the working variant.

    An HVDC line goes out with both converter stations and a tie line with both dangling lines, as neither is a
    connectable itself. A bus breaker bus loses everything on it. A busbar section goes out with the circuit breakers
    around it, whatever the contingency propagation of the security analysis is set to. Returns False when an element
    could not be disconnected.
    """
    types = outage_elements.types
    switches = [element for element in elements if types.get(element) == "SWITCH"]
    if switches:
        net.update_switches(pd.DataFrame({"open": True}, index=pd.Index(switches, name="id")))
    applied = True
    for element in elements:
        if element in switches:
            continue
        element_type = types.get(element)
        if element_type == "BUS":
            if not _disconnect_from_bus(net, element, outage_elements):
                applied = False
                logger.warning("powsybl_outage_not_applied", element=element)
            continue
        for part in outage_elements.parts.get(element, [element]):
            if not _disconnect(net, part, outage_elements):
                applied = False
                logger.warning("powsybl_outage_not_applied", element=element)
    return applied


def resolve_slack_buses(net: Network, slack_ids: Sequence[str]) -> list[str]:
    """Bus view ids for slack ids on the working variant.

    An injection id (e.g. the slack generator) stands for the bus it is connected to, so the slack follows the
    generator when an outage renumbers the buses. A bus breaker bus stands for the bus it is part of, and a bus view
    id for itself. Disconnected injections and ids that are none of these are left out, with a warning.
    """
    injection_buses = net.get_injections(attributes=["bus_id"])["bus_id"]
    bus_breaker_buses = net.get_bus_breaker_view_buses(attributes=["bus_id"])["bus_id"]
    bus_ids = set(net.get_buses(attributes=[]).index)
    resolved, unresolved = [], []
    for slack_id in slack_ids:
        if slack_id in injection_buses.index:
            bus_id = injection_buses[slack_id]
        elif slack_id in bus_breaker_buses.index:
            bus_id = bus_breaker_buses[slack_id]
        else:
            bus_id = slack_id if slack_id in bus_ids else ""
        if isinstance(bus_id, str) and bus_id:
            resolved.append(bus_id)
        else:
            unresolved.append(slack_id)
    if unresolved:
        # Without a slack to name the load flow parameters decide, which is not what was asked for
        logger.warning("powsybl_slack_override_not_connected", slack_ids=unresolved)
    return list(dict.fromkeys(resolved))


def with_slack_buses(
    lf_params: pypowsybl.loadflow.Parameters, slack_bus_ids: Sequence[str] | None, all_components: bool = False
) -> pypowsybl.loadflow.Parameters:
    """Copy *lf_params* to select the given buses as slack (by name) and, optionally, to solve all components."""
    params = deepcopy(lf_params)
    if all_components:
        params.component_mode = ComponentMode.ALL_CONNECTED
    if slack_bus_ids:
        # A slack terminal stored in the network would take precedence over the names
        params.read_slack_bus = False
        provider = dict(params.provider_parameters or {})
        provider.update({"slackBusSelectionMode": "NAME", "slackBusesIds": ",".join(slack_bus_ids)})
        params.provider_parameters = provider
    return params


def _collect_raw_rows(
    net: Network,
    contingency_id: str,
    monitored_branches: Sequence[str],
    monitored_trafo3w: Sequence[str],
    monitored_voltage_levels: Collection[str],
    slack_bus_ids: Sequence[str] | None,
    converged_components: Collection[int] | None = None,
) -> RawSecurityResults:
    """Read the solved working variant into raw security analysis rows for *contingency_id*.

    Busbar sections stand for the buses of node breaker voltage levels that have them, bus breaker view buses for the
    others. With explicit slack buses, only the islands holding one of them count as solved, as in a security analysis
    where islands without a slack are dropped. With *converged_components*, islands outside them have no rows.
    """
    voltage_levels = set(monitored_voltage_levels)
    buses = net.get_buses(attributes=["synchronous_component"])
    if slack_bus_ids:
        solved_components = set(buses["synchronous_component"].reindex(list(slack_bus_ids)).dropna().astype(int))
    else:
        solved_components = set(buses["synchronous_component"].dropna().astype(int))
    if converged_components is not None:
        solved_components &= set(converged_components)
    solved_buses = set(buses.index[buses["synchronous_component"].isin(solved_components)])

    busbar_sections = net.get_busbar_sections(attributes=["voltage_level_id", "bus_id", "v", "angle"])
    busbar_sections = busbar_sections[busbar_sections["voltage_level_id"].isin(voltage_levels)]
    with_sections = set(busbar_sections["voltage_level_id"])
    bus_breaker_buses = net.get_bus_breaker_view_buses(attributes=["voltage_level_id", "bus_id", "v_mag", "v_angle"])
    bus_breaker_buses = bus_breaker_buses[bus_breaker_buses["voltage_level_id"].isin(voltage_levels - with_sections)].rename(
        columns={"v_mag": "v", "v_angle": "angle"}
    )
    bus_rows = pd.concat([busbar_sections, bus_breaker_buses])
    bus_rows = bus_rows[bus_rows["bus_id"].isin(solved_buses) & bus_rows["v"].notna()]
    bus_frame = pl.DataFrame(
        {
            "contingency_id": contingency_id,
            "operator_strategy_id": "",
            "voltage_level_id": bus_rows["voltage_level_id"].astype(str).to_list(),
            "bus_id": bus_rows.index.astype(str).to_list(),
            "v_mag": bus_rows["v"].to_numpy(dtype=float),
            "v_angle": bus_rows["angle"].to_numpy(dtype=float),
        },
        schema=BUS_RESULT_SCHEMA,
    )
    bus_components = pl.DataFrame(
        {
            "contingency_id": contingency_id,
            "bus_id": bus_rows.index.astype(str).to_list(),
            "synchronous_component": bus_rows["bus_id"].map(buses["synchronous_component"]).to_numpy(dtype=int),
        },
        schema=BUS_COMPONENT_SCHEMA,
    )

    def branch_frame(frame: pd.DataFrame, columns: Sequence[str], id_column: str, schema: Mapping) -> pl.DataFrame:
        bus_columns = [column for column in frame.columns if column.startswith("bus") and column.endswith("_id")]
        unsolved = ~frame[bus_columns].isin(solved_buses).any(axis=1)
        # The security analysis reports no row for a branch it did not solve (tripped or in a dropped island)
        frame = frame[~unsolved & frame[columns].notna().any(axis=1)].copy()
        # A side that is off while another one is solved carries nothing, which the security analysis reports as 0
        for bus_column in bus_columns:
            side = bus_column.removeprefix("bus").removesuffix("_id")
            off = frame[bus_column].fillna("") == ""
            side_columns = [column for column in columns if column.endswith(side)]
            frame.loc[off, side_columns] = frame.loc[off, side_columns].fillna(0.0)
        data = {"contingency_id": contingency_id, id_column: frame.index.astype(str).to_list()}
        if "operator_strategy_id" in schema:
            data["operator_strategy_id"] = ""
        data.update({column: frame[column].to_numpy(dtype=float) for column in columns})
        return (
            pl.DataFrame(data)
            .with_columns(*[pl.lit(None, dtype=dtype).alias(name) for name, dtype in schema.items() if name not in data])
            .select(list(schema))
            .cast(dict(schema))
        )

    branch_columns = ["p1", "q1", "i1", "p2", "q2", "i2"]
    branches = net.get_branches(attributes=[*branch_columns, "bus1_id", "bus2_id"]).reindex(list(monitored_branches))
    three_windings_columns = ["p1", "q1", "i1", "p2", "q2", "i2", "p3", "q3", "i3"]
    transformers = net.get_3_windings_transformers(
        attributes=[*three_windings_columns, "bus1_id", "bus2_id", "bus3_id"]
    ).reindex(list(monitored_trafo3w))
    return RawSecurityResults(
        bus_frame.lazy(),
        branch_frame(branches, branch_columns, "branch_id", BRANCH_RESULT_SCHEMA).lazy(),
        branch_frame(transformers, three_windings_columns, "transformer_id", THREE_WINDINGS_RESULT_SCHEMA).lazy(),
        bus_components.lazy(),
    )


def _components_to_judge(
    net: Network, elements: Sequence[str], outage_elements: _OutageElements, slack_bus_ids: Sequence[str] | None
) -> set[int]:
    """Return the synchronous components of the working variant whose status is the status of the contingency.

    With given slacks these are the islands holding one, the only ones reported. Otherwise the main component and the
    islands reaching into a voltage level of an outaged element: an island the contingency does not touch that fails
    to solve (it does so in the base case too) says nothing about the contingency.
    """
    buses = net.get_buses(attributes=["synchronous_component", "voltage_level_id"])
    if slack_bus_ids:
        return set(buses["synchronous_component"].reindex(list(slack_bus_ids)).dropna().astype(int))
    touched = set().union(*(outage_elements.voltage_levels.get(element, set()) for element in elements))
    return {0, *buses.loc[buses["voltage_level_id"].isin(touched), "synchronous_component"].astype(int)}


def basecase_bus_rows(
    net: Network, lf_params: pypowsybl.loadflow.Parameters, monitored_voltage_levels: Collection[str]
) -> RawSecurityResults:
    """Return the bus rows of the base case with every island solved, under the contingency id of the base case.

    Contingencies solved one by one have no pre-contingency rows of their own; the deviation of a voltage from its
    base case value is read from these. Empty when the main component does not converge.
    """
    base_variant = net.get_working_variant_id()
    variant = f"basecase-{uuid.uuid4()}"
    net.clone_variant(base_variant, variant)
    net.set_working_variant(variant)
    try:
        component_results = pypowsybl.loadflow.run_ac(net, with_slack_buses(lf_params, None, all_components=True))
        if _overall_status(component_results, {0}) != ComponentStatus.CONVERGED:
            return RawSecurityResults.empty()
        converged = {
            result.synchronous_component_num for result in component_results if result.status == ComponentStatus.CONVERGED
        }
        return _collect_raw_rows(net, BASECASE_IN_RESULT, [], [], monitored_voltage_levels, None, converged)
    finally:
        net.set_working_variant(base_variant)
        net.remove_variant(variant)


def _elements_on_a_bus(net: Network) -> set[str]:
    """Return the branches, three winding transformers and injections with a side on a bus of the working variant."""
    frames = (
        net.get_branches(attributes=["bus1_id", "bus2_id"]),
        net.get_3_windings_transformers(attributes=["bus1_id", "bus2_id", "bus3_id"]),
        net.get_injections(attributes=["bus_id"]),
    )
    return {str(element) for frame in frames for element in frame.index[(frame.fillna("") != "").any(axis=1)]}


def run_contingencies_as_loadflows(
    net: Network,
    contingencies: Mapping[str, Sequence[str]],
    lf_params: pypowsybl.loadflow.Parameters,
    monitored_branches: Sequence[str],
    monitored_trafo3w: Sequence[str],
    monitored_voltage_levels: Collection[str],
    slack_bus_overrides: Mapping[str, Sequence[str]] | None = None,
    contingency_voltage_levels: Mapping[str, Collection[str]] | None = None,
) -> tuple[RawSecurityResults, dict[str, ComponentStatus], dict[str, list[str]]]:
    """Solve each contingency with its own AC load flow over every island.

    Each contingency is applied on a fresh variant of the working variant, which is left unchanged. Unlike the security
    analysis this solves every synchronous component, with the given slack buses where there are some.

    Parameters
    ----------
    net : Network
        The network in its base state.
    contingencies : Mapping[str, Sequence[str]]
        Contingency id to the ids of the elements it disconnects (switches are opened).
    lf_params : pypowsybl.loadflow.Parameters
        The load flow parameters of the security analysis; the component mode is set to all components.
    monitored_branches, monitored_trafo3w, monitored_voltage_levels
        What to report, as in the security analysis.
    slack_bus_overrides : Mapping[str, Sequence[str]] | None
        Contingency id to the slack of each island to solve, as injection ids (their bus after the outage) or bus ids.
        Islands without one of them are reported as not solved.
    contingency_voltage_levels : Mapping[str, Collection[str]] | None
        Contingency id to voltage levels reported in that contingency on top of the monitored ones.

    Returns
    -------
    tuple[RawSecurityResults, dict[str, ComponentStatus], dict[str, list[str]]]
        The raw rows of the converged contingencies, the status of every contingency and the elements every applied
        contingency disconnected: its own and the ones that lost their bus with them. The status is the worst of the
        islands the contingency is judged by (see ``_components_to_judge``); an island that does not converge has no
        rows.
    """
    slack_bus_overrides = slack_bus_overrides or {}
    contingency_voltage_levels = contingency_voltage_levels or {}
    base_variant = net.get_working_variant_id()
    outage_elements = _outage_elements(net, {element for elements in contingencies.values() for element in elements})
    on_a_bus = _elements_on_a_bus(net)
    results = []
    statuses: dict[str, ComponentStatus] = {}
    disconnected: dict[str, list[str]] = {}
    for contingency_id, elements in contingencies.items():
        variant = f"outage-{uuid.uuid4()}"
        net.clone_variant(base_variant, variant)
        net.set_working_variant(variant)
        try:
            if not _apply_outage(net, elements, outage_elements):
                statuses[contingency_id] = ComponentStatus.FAILED
                continue
            disconnected[contingency_id] = sorted({*elements, *(on_a_bus - _elements_on_a_bus(net))})
            slack_ids = slack_bus_overrides.get(contingency_id)
            slack_bus_ids = resolve_slack_buses(net, slack_ids) if slack_ids else None
            component_results = pypowsybl.loadflow.run_ac(
                net, with_slack_buses(lf_params, slack_bus_ids, all_components=True)
            )
            status = _overall_status(component_results, _components_to_judge(net, elements, outage_elements, slack_bus_ids))
            statuses[contingency_id] = status
            if status == ComponentStatus.CONVERGED:
                converged = {
                    result.synchronous_component_num
                    for result in component_results
                    if result.status == ComponentStatus.CONVERGED
                }
                results.append(
                    _collect_raw_rows(
                        net,
                        contingency_id,
                        monitored_branches,
                        monitored_trafo3w,
                        {*monitored_voltage_levels, *contingency_voltage_levels.get(contingency_id, ())},
                        slack_bus_ids,
                        converged,
                    )
                )
        finally:
            net.set_working_variant(base_variant)
            net.remove_variant(variant)
    return concat_raw_results(results), statuses, disconnected
