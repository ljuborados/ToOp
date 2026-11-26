# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import polars as pl
import pypowsybl
import pytest
from pypowsybl.loadflow import ComponentMode
from toop_engine_contingency_analysis.pypowsybl.contingency_analysis_powsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.outage_loadflows import (
    get_synchronous_components,
    resolve_slack_buses,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import get_single_slack_parameters
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition

ALL_COMPONENTS = pypowsybl.loadflow.Parameters(
    distributed_slack=False, component_mode=ComponentMode.ALL_CONNECTED, dc_use_transformer_ratio=True
)
SINGLE_SLACK = get_single_slack_parameters()


def _security(lf_params: pypowsybl.loadflow.Parameters) -> pypowsybl.security.Parameters:
    """The contingencies as they are defined, on one thread, with the given load flow parameters."""
    return pypowsybl.security.Parameters(
        load_flow_parameters=lf_params, provider_parameters={"threadCount": "1", "contingencyPropagation": "false"}
    )


def _definition(net: pypowsybl.network.Network, contingencies: list[Contingency]) -> Nminus1Definition:
    branches = [
        MonitoredElement(id=branch_id, name="", kind="branch", type=row.type)
        for branch_id, row in net.get_branches(attributes=["type"]).iterrows()
    ]
    buses = [
        MonitoredElement(id=bus_id, name="", kind="bus", type="BUS") for bus_id in net.get_bus_breaker_view_buses().index
    ]
    return Nminus1Definition(
        monitored_elements=[*branches, *buses],
        contingencies=[Contingency(id="BASECASE", elements=[]), *contingencies],
    )


def _outage(element_id: str, kind: str = "branch") -> Contingency:
    return Contingency(id=element_id, elements=[GridElement(id=element_id, name="", kind=kind, type=None)])


def _branch_i(results: LoadflowResultsPolars, contingency: str) -> dict[tuple[str, int], float]:
    frame = results.branch_results.collect().filter(pl.col("contingency") == contingency)
    return {(row["element"], row["side"]): row["i"] for row in frame.iter_rows(named=True)}


def _island_ieee14() -> pypowsybl.network.Network:
    """IEEE 14 with buses 7 and 8 (a transformer and the condenser) cut off from the rest: a second component."""
    net = pypowsybl.network.create_ieee14()
    net.disconnect("T4-7-1")
    net.disconnect("L7-9-1")
    return net


def test_empty_contingency_other_than_the_basecase_gets_the_basecase_values() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [Contingency(id="nothing", elements=[]), _outage("L2-3-1")])

    res = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )

    converged = dict(res.converged.collect().select("contingency", "status").iter_rows())
    assert converged == {"BASECASE": "CONVERGED", "nothing": "CONVERGED", "L2-3-1": "CONVERGED"}
    assert _branch_i(res, "nothing") == _branch_i(res, "BASECASE")
    assert _branch_i(res, "L2-3-1") != _branch_i(res, "BASECASE")


def test_contingency_without_impact_gets_the_basecase_values() -> None:
    net = pypowsybl.network.create_ieee14()
    net.disconnect("L1-5-1")
    definition = _definition(net, [_outage("L1-5-1")])

    default = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )
    assert dict(default.converged.collect().select("contingency", "status").iter_rows())["L1-5-1"] == "NO_CALCULATION"

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        fill_no_impact_contingencies=True,
    )

    converged = dict(res.converged.collect().select("contingency", "status").iter_rows())
    assert converged["L1-5-1"] == "CONVERGED"
    base, outage = _branch_i(res, "BASECASE"), _branch_i(res, "L1-5-1")
    assert base.keys() == outage.keys()
    assert all(np.isclose(base[key], outage[key], equal_nan=True) for key in base)


def test_non_main_components_are_reported_only_when_asked() -> None:
    net = _island_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])

    def island_rows(fill: bool) -> pl.DataFrame:
        res = run_contingency_analysis_powsybl(
            net,
            definition,
            job_id="j",
            timestep=0,
            polars=True,
            security_analysis_params=_security(ALL_COMPONENTS),
            fill_non_main_components=fill,
        )
        nodes = res.node_results.collect()
        return nodes.filter(pl.col("element").is_in(["B7", "B8"]) & pl.col("vm").is_not_nan())

    without = island_rows(fill=False)
    assert set(without["contingency"]) == {"BASECASE"}, "the security analysis solves the main component only"

    with_fill = island_rows(fill=True)
    assert set(with_fill["contingency"]) == {"BASECASE", "L2-3-1"}
    base = with_fill.filter(pl.col("contingency") == "BASECASE").sort("element")
    outage = with_fill.filter(pl.col("contingency") == "L2-3-1").sort("element")
    np.testing.assert_allclose(base["vm"].to_numpy(), outage["vm"].to_numpy())


def test_contingency_inside_a_non_main_component_is_solved_with_its_own_loadflow() -> None:
    net = _island_ieee14()
    components = get_synchronous_components(net)
    assert components.branch["L7-8-1"] != 0
    definition = _definition(net, [_outage("L7-8-1")])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(ALL_COMPONENTS),
        fill_non_main_components=True,
    )

    converged = dict(res.converged.collect().select("contingency", "status").iter_rows())
    assert converged["L7-8-1"] == "CONVERGED"
    # The main component is untouched by this contingency, so its values are the base case ones
    base, outage = _branch_i(res, "BASECASE"), _branch_i(res, "L7-8-1")
    assert np.isclose(base[("L1-2-1", 1)], outage[("L1-2-1", 1)])


def test_loadflow_path_matches_the_security_analysis() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1"), _outage("L6-13-1")])

    by_analysis = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )
    by_loadflow = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"L2-3-1"},
    )

    expected, actual = _branch_i(by_analysis, "L2-3-1"), _branch_i(by_loadflow, "L2-3-1")
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        assert np.isclose(value, actual[key], rtol=1e-6, atol=1e-6, equal_nan=True), key
    nodes = by_loadflow.node_results.collect().filter(pl.col("contingency") == "L2-3-1")
    assert nodes.height == len(net.get_bus_breaker_view_buses())


def test_run_contingency_analysis_powsybl_solves_a_contingency_with_its_slack_override() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])
    lf_params = pypowsybl.loadflow.Parameters(distributed_slack=False, dc_use_transformer_ratio=True, read_slack_bus=False)

    default = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(lf_params)
    )
    moved = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(lf_params),
        slack_overrides={"L2-3-1": ["B8-G"]},
    )
    reference_params = pypowsybl.loadflow.Parameters(
        distributed_slack=False,
        dc_use_transformer_ratio=True,
        read_slack_bus=False,
        provider_parameters={"slackBusSelectionMode": "NAME", "slackBusesIds": net.get_generators().loc["B8-G", "bus_id"]},
    )
    reference = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(reference_params)
    )
    assert _branch_i(moved, "BASECASE") == pytest.approx(_branch_i(default, "BASECASE"), nan_ok=True)
    assert _branch_i(moved, "L2-3-1") == pytest.approx(_branch_i(reference, "L2-3-1"), nan_ok=True)
    assert _branch_i(moved, "L2-3-1") != pytest.approx(_branch_i(default, "L2-3-1"), nan_ok=True)


def _switch_definition(switch_id: str, contingencies: list[Contingency]) -> Nminus1Definition:
    return Nminus1Definition(
        monitored_elements=[MonitoredElement(id=switch_id, name="", kind="switch", type="SWITCH")],
        contingencies=[Contingency(id="BASECASE", elements=[]), *contingencies],
    )


@pytest.mark.parametrize("breaker", ["S1VL1_LD1_BREAKER", "S3VL1_LINES2S3_BREAKER"])
def test_run_contingency_analysis_powsybl_has_no_angle_through_a_monitored_switch_its_contingency_opens(
    breaker: str,
) -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    opened = Contingency(id="open", elements=[GridElement(id=breaker, name="", kind="switch", type=None)])

    res = run_contingency_analysis_powsybl(
        net,
        _switch_definition(breaker, [opened]),
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
    )

    va_diff = res.va_diff_results.collect()
    assert va_diff.filter(pl.col("contingency") == "BASECASE").height == 0, "the switch is closed in the base case"
    row = va_diff.filter((pl.col("contingency") == "open") & (pl.col("element") == breaker))
    assert row.height == 1
    # The feeder side reaches no busbar section without the breaker, so there is no angle to read for it. Mapped
    # through the breaker itself, both sides would be one bus and the difference exactly 0.
    assert row["va_diff"][0] is None or np.isnan(row["va_diff"][0])


def _statuses(results: LoadflowResultsPolars) -> dict[str, str]:
    return dict(results.converged.collect().select("contingency", "status").iter_rows())


def _group(contingency_id: str, element_ids: list[str]) -> Contingency:
    return Contingency(
        id=contingency_id,
        elements=[GridElement(id=element_id, name="", kind="branch", type=None) for element_id in element_ids],
    )


@pytest.mark.parametrize(
    "element_ids",
    [["S1VL1_LD1_BREAKER", "LD1"], ["S2VL1_LINES2S3_BREAKER", "S3VL1_LINES2S3_BREAKER", "LINE_S2S3"]],
)
def test_loadflow_path_takes_an_element_its_own_breakers_isolate(element_ids: list[str]) -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    definition = _definition(net, [_group("c", element_ids)])

    by_analysis = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )
    by_loadflow = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"c"},
    )

    assert _statuses(by_loadflow)["c"] == "CONVERGED"
    # The security analysis solves the main synchronous component only, the load flow every island
    expected, actual = _branch_i(by_analysis, "c"), _branch_i(by_loadflow, "c")
    assert expected and expected.keys() <= actual.keys()
    assert {key: actual[key] for key in expected} == pytest.approx(expected, rel=1e-3, abs=0.1, nan_ok=True)


def test_loadflow_path_takes_an_element_that_is_off_already() -> None:
    net = pypowsybl.network.create_ieee14()
    net.disconnect("L1-5-1")
    definition = _definition(net, [_outage("L1-5-1")])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"L1-5-1"},
    )

    assert _statuses(res)["L1-5-1"] == "CONVERGED"


def test_loadflow_path_takes_an_hvdc_line_out_with_its_converter_stations() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    definition = _definition(net, [_group("HVDC1", ["HVDC1"]), _group("LINE_S2S3", ["LINE_S2S3"])])

    by_analysis = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )
    by_loadflow = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"HVDC1"},
    )

    assert _statuses(by_loadflow) == {"BASECASE": "CONVERGED", "HVDC1": "CONVERGED", "LINE_S2S3": "CONVERGED"}
    expected, actual = _branch_i(by_analysis, "HVDC1"), _branch_i(by_loadflow, "HVDC1")
    assert expected and expected.keys() <= actual.keys()
    assert {key: actual[key] for key in expected} == pytest.approx(expected, rel=1e-3, abs=1e-2, nan_ok=True)


def test_loadflow_path_reports_the_deviation_from_the_base_case_voltage() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])

    def deviations(**kwargs: object) -> dict[str, float]:
        res = run_contingency_analysis_powsybl(
            net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK), **kwargs
        )
        nodes = res.node_results.collect().filter(pl.col("contingency") == "L2-3-1")
        return dict(nodes.select("element", "vm_basecase_deviation").iter_rows())

    by_loadflow = deviations(loadflow_contingency_ids={"L2-3-1"})
    assert not any(np.isnan(value) for value in by_loadflow.values())
    assert by_loadflow == pytest.approx(deviations(), abs=1e-3)


def test_loadflow_path_has_no_angle_difference_between_two_islands() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    coupler = "S1VL2_COUPLER"
    definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=coupler, name="", kind="switch", type="SWITCH")],
        contingencies=[
            Contingency(id="BASECASE", elements=[]),
            Contingency(id="open", elements=[GridElement(id=coupler, name="", kind="switch", type=None)]),
        ],
    )

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"open"},
    )

    row = res.va_diff_results.collect().filter((pl.col("contingency") == "open") & (pl.col("element") == coupler))
    assert row.height == 1
    assert row["va_diff"][0] is None or np.isnan(row["va_diff"][0])


def test_loadflow_path_is_not_failed_by_an_island_it_does_not_touch() -> None:
    net = _island_ieee14()
    # The island cannot serve this load, so its load flow fails in the base case as well
    net.create_loads(id="TOO_MUCH", voltage_level_id="VL7", bus_id="B7", p0=5000.0, q0=3000.0)
    definition = _definition(net, [_outage("L2-3-1")])

    by_analysis = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )
    by_loadflow = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"L2-3-1"},
    )

    assert _statuses(by_loadflow)["L2-3-1"] == "CONVERGED"
    expected, actual = _branch_i(by_analysis, "L2-3-1"), _branch_i(by_loadflow, "L2-3-1")
    main = [key for key in expected if not np.isnan(expected[key])]
    assert main
    assert {key: actual[key] for key in main} == pytest.approx({key: expected[key] for key in main}, rel=1e-4, abs=1e-3)


def test_filled_components_do_not_double_the_rows_the_analysis_reports() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    definition = _definition(net, [_group("HVDC1", ["HVDC1"]), _group("HVDC2", ["HVDC2"])])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        fill_non_main_components=True,
    )

    branches = res.branch_results.collect()
    assert branches.height == branches.unique(subset=["contingency", "element", "side"]).height


def test_filling_non_main_components_needs_ac() -> None:
    net = _island_ieee14()
    with pytest.raises(ValueError, match="method='ac'"):
        run_contingency_analysis_powsybl(
            net, _definition(net, []), job_id="j", timestep=0, method="dc", polars=True, fill_non_main_components=True
        )


@pytest.mark.parametrize("also_retained", [False, True])
def test_a_contingency_can_open_a_monitored_switch_that_is_not_retained(also_retained: bool) -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    disconnector = "S1VL2_BBS1_GH1_DISCONNECTOR"
    switches = [disconnector, *(["S1VL2_GH1_BREAKER"] if also_retained else [])]
    definition = Nminus1Definition(
        monitored_elements=[MonitoredElement(id=switch, name="", kind="switch", type="SWITCH") for switch in switches],
        contingencies=[
            Contingency(id="BASECASE", elements=[]),
            Contingency(
                id="open", elements=[GridElement(id=switch, name="", kind="switch", type=None) for switch in switches]
            ),
        ],
    )

    res = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )

    assert _statuses(res)["open"] == "CONVERGED"
    rows = res.va_diff_results.collect().filter((pl.col("contingency") == "open") & (pl.col("element") == disconnector))
    assert rows.height == 1
    assert rows["va_diff"][0] is None or np.isnan(rows["va_diff"][0]), "the two sides are one bus breaker bus"


def test_resolve_slack_buses_returns_the_bus_of_an_injection_and_of_a_bus_breaker_bus() -> None:
    net = pypowsybl.network.create_ieee14()

    assert resolve_slack_buses(net, ["B8-G", "B8", "VL3_0", "unknown"]) == ["VL8_0", "VL3_0"]


def test_run_contingency_analysis_powsybl_takes_a_bus_breaker_bus_as_slack_override() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])
    lf_params = pypowsybl.loadflow.Parameters(distributed_slack=False, dc_use_transformer_ratio=True, read_slack_bus=False)

    def currents(slack_id: str, **kwargs: object) -> dict[tuple[str, int], float]:
        res = run_contingency_analysis_powsybl(
            net,
            definition,
            job_id="j",
            timestep=0,
            polars=True,
            security_analysis_params=_security(lf_params),
            slack_overrides={"L2-3-1": [slack_id]},
            **kwargs,
        )
        assert _statuses(res)["L2-3-1"] == "CONVERGED"
        return _branch_i(res, "L2-3-1")

    assert currents("B8") == pytest.approx(currents("B8-G"), nan_ok=True)
    assert currents("B8", loadflow_contingency_ids={"L2-3-1"}) == pytest.approx(currents("B8-G"), rel=1e-4, nan_ok=True)


def _matches_the_security_analysis(
    net: pypowsybl.network.Network, contingency: Contingency, propagation: str = "false"
) -> None:
    """Assert that *contingency* solved with its own load flow gives the currents of the security analysis.

    Both solve every island: an HVDC link into an island that is not solved is modelled differently.
    """
    params = pypowsybl.security.Parameters(
        load_flow_parameters=ALL_COMPONENTS, provider_parameters={"threadCount": "1", "contingencyPropagation": propagation}
    )
    definition = _definition(net, [contingency])
    by_analysis = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=params
    )
    by_loadflow = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=params,
        loadflow_contingency_ids={contingency.id},
    )
    assert _statuses(by_analysis)[contingency.id] == "CONVERGED"
    assert _statuses(by_loadflow)[contingency.id] == "CONVERGED"
    expected, actual = _branch_i(by_analysis, contingency.id), _branch_i(by_loadflow, contingency.id)
    solved = {key: value for key, value in expected.items() if not np.isnan(value)}
    assert solved
    assert {key: actual[key] for key in solved} == pytest.approx(solved, rel=1e-3, abs=0.1)


def test_run_contingencies_as_loadflows_takes_an_element_out_behind_a_fictitious_breaker() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    net.update_switches(id="S2VL1_LINES2S3_BREAKER", fictitious=True)

    _matches_the_security_analysis(net, _group("LINE_S2S3", ["LINE_S2S3"]))


def test_run_contingencies_as_loadflows_takes_a_bus_breaker_bus_out_with_everything_on_it() -> None:
    net = pypowsybl.network.create_ieee14()

    _matches_the_security_analysis(net, _outage("B3", kind="bus"))


def test_run_contingencies_as_loadflows_takes_a_busbar_section_out_with_the_breakers_around_it() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()

    _matches_the_security_analysis(net, _outage("S2VL1_BBS", kind="bus"), propagation="true")


def test_run_contingency_analysis_powsybl_solves_a_bus_contingency_inside_a_non_main_component() -> None:
    net = _island_ieee14()
    definition = _definition(net, [_outage("B8", kind="bus")])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(ALL_COMPONENTS),
        fill_non_main_components=True,
    )

    nodes = res.node_results.collect().filter(pl.col("element").is_in(["B7", "B8"]))
    base = nodes.filter(pl.col("contingency") == "BASECASE")
    assert base.height == 2 and not base["vm"].is_nan().any()
    # B8 holds the only source of the island, so without it the island has no voltage
    outage = nodes.filter(pl.col("contingency") == "B8")
    assert outage["vm"].is_nan().all() or outage["vm"].is_null().all()


@pytest.mark.parametrize("security_analysis_params", [None, _security(SINGLE_SLACK)])
def test_run_contingency_analysis_powsybl_fills_non_main_components_whatever_the_component_mode(
    security_analysis_params: pypowsybl.security.Parameters | None,
) -> None:
    net = _island_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=security_analysis_params,
        fill_non_main_components=True,
    )

    nodes = res.node_results.collect().filter(pl.col("element").is_in(["B7", "B8"]))
    for contingency in ("BASECASE", "L2-3-1"):
        rows = nodes.filter(pl.col("contingency") == contingency)
        assert rows.height == 2, contingency
        assert not (rows["vm"].is_nan() | rows["vm"].is_null()).any(), contingency
    island_branch = res.branch_results.collect().filter(pl.col("element") == "L7-8-1")
    assert set(island_branch["contingency"]) == {"BASECASE", "L2-3-1"}


def test_run_contingency_analysis_powsybl_reports_the_base_case_deviation_of_contingency_voltage_levels() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1")])
    extra = ["VL12", "VL13", "VL14"]

    def deviations(**kwargs: object) -> dict[str, float]:
        res = run_contingency_analysis_powsybl(
            net,
            definition,
            job_id="j",
            timestep=0,
            polars=True,
            security_analysis_params=_security(SINGLE_SLACK),
            monitored_voltage_levels=["VL1", "VL2", "VL3"],
            contingency_voltage_levels={"L2-3-1": extra},
            **kwargs,
        )
        nodes = res.node_results.collect().filter(
            (pl.col("contingency") == "L2-3-1") & pl.col("element").is_in(["B12", "B13", "B14"])
        )
        return dict(nodes.select("element", "vm_basecase_deviation").iter_rows())

    by_analysis = deviations()
    assert len(by_analysis) == 3
    assert not any(value is None or np.isnan(value) for value in by_analysis.values())
    assert by_analysis == pytest.approx(deviations(loadflow_contingency_ids={"L2-3-1"}), abs=1e-3)


def test_run_contingency_analysis_powsybl_lists_what_a_loadflow_contingency_disconnects() -> None:
    net = pypowsybl.network.create_ieee14()
    definition = _definition(net, [_outage("L2-3-1"), _outage("L1-5-1")])

    res = run_contingency_analysis_powsybl(
        net,
        definition,
        job_id="j",
        timestep=0,
        polars=True,
        security_analysis_params=_security(SINGLE_SLACK),
        loadflow_contingency_ids={"L2-3-1"},
    )

    connectivity = res.connectivity_result.collect().sort("contingency")
    assert connectivity.select("contingency", "element", "outage_group_id").rows() == [
        ("L1-5-1", "L1-5-1", "L1-5-1"),
        ("L2-3-1", "L2-3-1", "L2-3-1"),
    ]


def test_run_contingency_analysis_powsybl_gives_an_empty_contingency_the_base_case_angle_differences() -> None:
    net = pypowsybl.network.create_four_substations_node_breaker_network()
    coupler = "S1VL2_COUPLER"
    net.update_switches(id=coupler, open=True)
    definition = _switch_definition(coupler, [Contingency(id="nothing", elements=[]), _group("LINE_S3S4", ["LINE_S3S4"])])

    res = run_contingency_analysis_powsybl(
        net, definition, job_id="j", timestep=0, polars=True, security_analysis_params=_security(SINGLE_SLACK)
    )

    va_diff = res.va_diff_results.collect().filter(pl.col("element") == coupler)
    by_contingency = dict(va_diff.select("contingency", "va_diff").iter_rows())
    assert set(by_contingency) == {"BASECASE", "nothing", "LINE_S3S4"}
    assert by_contingency["nothing"] == pytest.approx(by_contingency["BASECASE"], nan_ok=True)
