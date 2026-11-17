# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pypowsybl
import pytest
from toop_engine_grid_helpers.powsybl.outage_groups import get_node_breaker_graph, get_outage_groups


@pytest.fixture
def network() -> pypowsybl.network.Network:
    return pypowsybl.network.create_four_substations_node_breaker_network()


def _terminal_buses(network: pypowsybl.network.Network) -> dict[str, set[str]]:
    buses: dict[str, set[str]] = {}
    for element_id, bus_id in network.get_terminals(attributes=["bus_id"])["bus_id"].items():
        buses.setdefault(element_id, set()).add(bus_id)
    return buses


def test_graph_numbers_nodes_network_wide(network: pypowsybl.network.Network) -> None:
    graph = get_node_breaker_graph(network)
    assert graph.n_nodes == len(graph.node_voltage_level)
    assert graph.switches[["node1", "node2"]].to_numpy().max() < graph.n_nodes
    assert set(graph.node_voltage_level) == set(network.get_voltage_levels().index)
    assert set(network.get_lines().index) <= set(graph.terminals["connectable_id"])


@pytest.mark.parametrize("element_id", ["LINE_S2S3", "LINE_S3S4", "TWT", "GH1", "LD1", "S1VL2_BBS1"])
def test_opening_the_breakers_isolates_the_group(network: pypowsybl.network.Network, element_id: str) -> None:
    groups = get_outage_groups(network, [element_id])
    group = groups.group_of_element[element_id]
    assert element_id in groups.elements[group]
    switches = network.get_switches()
    assert all(
        switches.at[breaker, "kind"] == "BREAKER" and not switches.at[breaker, "open"] for breaker in groups.breakers[group]
    )

    network.update_switches(id=groups.breakers[group], open=[True] * len(groups.breakers[group]))
    buses = _terminal_buses(network)
    for member in groups.elements[group]:
        # every element of the group is dead or only touches buses without anything outside the group
        assert buses[member] <= {""} | {bus for other in groups.elements[group] for bus in buses[other]}
    outside = set(buses) - set(groups.elements[group])
    group_buses = {bus for member in groups.elements[group] for bus in buses[member]} - {""}
    assert not any(buses[element] & group_buses for element in outside)


def test_lines_join_their_terminals_unless_excluded(network: pypowsybl.network.Network) -> None:
    joined = get_outage_groups(network, ["LINE_S2S3"])
    split = get_outage_groups(network, ["LINE_S2S3"], non_propagating_ids=["LINE_S2S3"])
    group_vls = set(
        get_node_breaker_graph(network).node_voltage_level.iloc[split.nodes[split.group_of_element["LINE_S2S3"]]]
    )
    assert len(group_vls) == 1
    joined_vls = set(
        get_node_breaker_graph(network).node_voltage_level.iloc[joined.nodes[joined.group_of_element["LINE_S2S3"]]]
    )
    assert joined_vls == {"S2VL1", "S3VL1"}


def test_switches_form_their_own_group_and_unknown_ids_are_left_out(network: pypowsybl.network.Network) -> None:
    switch_id = network.get_switches().query("kind == 'BREAKER'").index[0]
    groups = get_outage_groups(network, [switch_id, "LINE_S3S4", "NOT_THERE"])
    group = groups.group_of_element[switch_id]
    assert groups.elements[group] == [switch_id]
    assert groups.breakers[group] == []
    assert group != groups.group_of_element["LINE_S3S4"]
    assert "NOT_THERE" not in groups.group_of_element


def test_outage_lists_breakers_then_elements(network: pypowsybl.network.Network) -> None:
    groups = get_outage_groups(network, ["LINE_S3S4", "LD1"])
    group_ids, ids = groups.outage(["LINE_S3S4", "LD1"])
    assert len(group_ids) == 2
    breakers = {breaker for group in group_ids for breaker in groups.breakers[group]}
    assert ids[: len(breakers)] == sorted(breakers)
    assert {"LINE_S3S4", "LD1"} <= set(ids)
    assert np.array_equal(np.sort(groups.nodes[group_ids[0]]), groups.nodes[group_ids[0]])
