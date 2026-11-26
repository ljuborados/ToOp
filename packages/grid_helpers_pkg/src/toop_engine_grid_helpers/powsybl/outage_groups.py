# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Outage groups on the node-breaker topology of a powsybl network.

An outage group is what a protection scheme has to switch off to isolate a faulted element: everything connected to
it without passing a circuit breaker. Nodes are joined by internal connections and by closed switches that are not
breakers, and the propagating branches (lines and transformers by default) join the nodes of their terminals. The
group is cleared by opening the closed breakers that touch it, which disconnects every element on its nodes.

The groups only depend on the topology, so they are computed once for all contingencies.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from pypowsybl.network import Network

# Connectable types whose terminals belong to one outage group
PROPAGATING_CONNECTABLE_TYPES = ("LINE", "TWO_WINDINGS_TRANSFORMER", "THREE_WINDINGS_TRANSFORMER", "TIE_LINE")


@dataclass
class NodeBreakerGraph:
    """The node-breaker topology of all voltage levels, with nodes numbered network wide."""

    terminals: pd.DataFrame
    """One row per terminal: ``connectable_id``, ``connectable_type``, ``node`` (network wide)."""
    switches: pd.DataFrame
    """Indexed by switch id: ``kind``, ``open``, ``voltage_level_id``, ``node1``, ``node2`` (network wide)."""
    internal_connections: np.ndarray
    """Pairs of network wide nodes, shape (n, 2)."""
    node_voltage_level: pd.Series
    """Voltage level id per network wide node."""

    @property
    def n_nodes(self) -> int:
        """The number of network wide nodes."""
        return len(self.node_voltage_level)


def get_node_breaker_graph(network: Network) -> NodeBreakerGraph:
    """Collect the node-breaker topology of every node-breaker voltage level.

    Parameters
    ----------
    network : Network
        The network. Bus-breaker voltage levels are skipped.

    Returns
    -------
    NodeBreakerGraph
        Terminals, switches and internal connections with nodes numbered network wide.
    """
    voltage_levels = network.get_voltage_levels(attributes=["topology_kind"])
    node_breaker_levels = voltage_levels.index[voltage_levels["topology_kind"] == "NODE_BREAKER"]
    terminal_frames, connection_frames, offsets, level_of_node = [], [], {}, []
    offset = 0
    for voltage_level_id in node_breaker_levels:
        topology = network.get_node_breaker_topology(voltage_level_id)
        nodes = topology.nodes
        # A voltage level without anything in it has no nodes
        n_nodes = (
            int(
                max(
                    nodes.index.max() if len(nodes) else -1,
                    topology.switches[["node1", "node2"]].to_numpy().max(initial=-1),
                    topology.internal_connections[["node1", "node2"]].to_numpy().max(initial=-1),
                )
            )
            + 1
        )
        offsets[voltage_level_id] = offset
        level_of_node.append(np.full(n_nodes, voltage_level_id, dtype=object))
        terminals = nodes[nodes["connectable_id"] != ""]
        terminal_frames.append(
            pd.DataFrame(
                {
                    "connectable_id": terminals["connectable_id"].to_numpy(),
                    "connectable_type": terminals["connectable_type"].to_numpy(),
                    "node": terminals.index.to_numpy() + offset,
                }
            )
        )
        if len(topology.internal_connections):
            connection_frames.append(topology.internal_connections[["node1", "node2"]].to_numpy() + offset)
        offset += n_nodes

    switches = network.get_switches(attributes=["kind", "open", "voltage_level_id", "node1", "node2"])
    switches = switches[switches["voltage_level_id"].isin(offsets)].copy()
    # Explicit integers: without any node-breaker voltage level the columns are empty and have no integer type
    shift = switches["voltage_level_id"].map(offsets).to_numpy(dtype=np.int64)
    switches["node1"] = switches["node1"].to_numpy(dtype=np.int64) + shift
    switches["node2"] = switches["node2"].to_numpy(dtype=np.int64) + shift
    terminals = (
        pd.concat(terminal_frames, ignore_index=True)
        if terminal_frames
        else pd.DataFrame(
            {
                "connectable_id": pd.Series(dtype=object),
                "connectable_type": pd.Series(dtype=object),
                "node": pd.Series(dtype=np.int64),
            }
        )
    )
    connections = np.concatenate(connection_frames) if connection_frames else np.empty((0, 2), dtype=int)
    node_voltage_level = pd.Series(np.concatenate(level_of_node) if level_of_node else np.array([], dtype=object))
    return NodeBreakerGraph(
        terminals=terminals, switches=switches, internal_connections=connections, node_voltage_level=node_voltage_level
    )


@dataclass
class OutageGroups:
    """The outage group of each element and what clearing each group switches off."""

    group_of_element: dict[str, int]
    """Element id -> group id. Switches form their own group (their outage only opens them)."""
    elements: dict[int, list[str]] = field(default_factory=dict)
    """Group id -> the connectables with a terminal on the group, or the switch of a switch group."""
    breakers: dict[int, list[str]] = field(default_factory=dict)
    """Group id -> the closed breakers that touch the group and are opened to clear it."""
    nodes: dict[int, np.ndarray] = field(default_factory=dict)
    """Group id -> the network wide nodes of the group (empty for switch groups)."""
    node_labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """The group id every network wide node would belong to (see ``NodeBreakerGraph``)."""

    def outage(self, element_ids: Iterable[str]) -> tuple[list[int], list[str]]:
        """Return the groups of *element_ids* and the ids to switch off for them: closed breakers, then elements."""
        groups = sorted({self.group_of_element[element_id] for element_id in element_ids})
        breakers = sorted({breaker for group in groups for breaker in self.breakers[group]})
        elements = sorted({element for group in groups for element in self.elements[group]} - set(breakers))
        return groups, breakers + elements


def _node_components(graph: NodeBreakerGraph, propagating: pd.DataFrame) -> np.ndarray:
    """Component label of every network wide node."""
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.n_nodes))
    nx_graph.add_edges_from(map(tuple, graph.internal_connections))
    switches = graph.switches
    joining = switches[(switches["kind"] != "BREAKER") & ~switches["open"]]
    nx_graph.add_edges_from(zip(joining["node1"], joining["node2"], strict=True))
    for _, terminal_nodes in propagating.groupby("connectable_id")["node"]:
        first, *others = terminal_nodes.to_numpy()
        nx_graph.add_edges_from((first, node) for node in others)
    labels = np.empty(graph.n_nodes, dtype=np.int64)
    for label, component in enumerate(nx.connected_components(nx_graph)):
        labels[list(component)] = label
    return labels


def get_outage_groups(
    network: Network,
    element_ids: Collection[str],
    graph: NodeBreakerGraph | None = None,
    non_propagating_ids: Collection[str] = (),
    propagating_types: Collection[str] = PROPAGATING_CONNECTABLE_TYPES,
) -> OutageGroups:
    """Compute the outage groups of *element_ids* on the node-breaker topology.

    Parameters
    ----------
    network : Network
        The network in its base topology.
    element_ids : Collection[str]
        Connectables, busbar sections or switches to outage. Unknown ids are left out of the result.
    graph : NodeBreakerGraph | None
        The topology of *network*, if already collected.
    non_propagating_ids : Collection[str]
        Connectables of a propagating type that still do not join their terminals, e.g. lines that stand for network
        equivalents.
    propagating_types : Collection[str]
        Connectable types whose terminals belong to one group.

    Returns
    -------
    OutageGroups
        The group of every known element, and per group its elements and the closed breakers to open.
    """
    graph = graph if graph is not None else get_node_breaker_graph(network)
    terminals = graph.terminals
    propagating = terminals[
        terminals["connectable_type"].isin(list(propagating_types))
        & ~terminals["connectable_id"].isin(list(non_propagating_ids))
    ]
    labels = _node_components(graph, propagating)

    switches = graph.switches
    first_node = terminals.drop_duplicates("connectable_id").set_index("connectable_id")["node"]
    requested = pd.Index(list(dict.fromkeys(element_ids)))
    on_nodes = requested[requested.isin(first_node.index)]
    group_of_element = dict(zip(on_nodes, labels[first_node.loc[on_nodes].to_numpy(dtype=np.int64)].tolist(), strict=True))

    groups = sorted(set(group_of_element.values()))
    terminal_groups = labels[terminals["node"].to_numpy(dtype=np.int64)]
    in_groups = np.isin(terminal_groups, groups)
    elements = (
        pd.DataFrame(
            {"group": terminal_groups[in_groups], "connectable_id": terminals["connectable_id"].to_numpy()[in_groups]}
        )
        .drop_duplicates()
        .groupby("group")["connectable_id"]
        .agg(sorted)
        .to_dict()
    )
    closed_breakers = switches[(switches["kind"] == "BREAKER") & ~switches["open"]]
    breaker_rows = pd.concat(
        [
            pd.DataFrame({"group": labels[closed_breakers[column].to_numpy()], "switch_id": closed_breakers.index})
            for column in ("node1", "node2")
        ]
    )
    breaker_rows = breaker_rows[breaker_rows["group"].isin(groups)].drop_duplicates()
    breakers = breaker_rows.groupby("group")["switch_id"].agg(sorted).to_dict()
    nodes = {group: np.flatnonzero(labels == group) for group in groups} if groups else {}
    result = OutageGroups(
        group_of_element=group_of_element,
        elements={group: elements.get(group, []) for group in groups},
        breakers={group: breakers.get(group, []) for group in groups},
        nodes=nodes,
        node_labels=labels,
    )

    next_group = int(labels.max(initial=-1)) + 1
    for switch_id in requested[requested.isin(switches.index) & ~requested.isin(first_node.index)]:
        result.group_of_element[switch_id] = next_group
        result.elements[next_group] = [switch_id]
        result.breakers[next_group] = []
        result.nodes[next_group] = np.empty(0, dtype=np.int64)
        next_group += 1
    return result
