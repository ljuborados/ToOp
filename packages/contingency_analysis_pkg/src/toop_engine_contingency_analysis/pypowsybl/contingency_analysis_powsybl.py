# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Compute the N-1 AC/DC power flow for the network."""

from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import polars as pl
import pypowsybl
from beartype.typing import Literal, Optional, Sequence, Union
from pypowsybl.network import Network
from pypowsybl.security import SecurityAnalysisResult
from toop_engine_contingency_analysis.pypowsybl.outage_loadflows import (
    RawSecurityResults,
    SynchronousComponents,
    basecase_bus_rows,
    concat_raw_results,
    conform_raw_results,
    contingencies_touching_non_main_components,
    copy_basecase_rows,
    get_synchronous_components,
    resolve_slack_buses,
    run_contingencies_as_loadflows,
    with_slack_buses,
)
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import (
    PowsyblNMinus1Definition,
    add_name_column,
    convert_powsybl_status,
    fingerprint_current_limits_for_powsybl,
    get_convergence_result_df,
    get_current_branch_limits_for_powsybl,
    get_regulating_element_results,
    normalize_monitored_branches_for_powsybl,
    prepare_branch_limits,
    translate_nminus1_for_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers_polars import (
    add_name_column_polars,
    get_branch_limits_polars,
    get_branch_results_polars,
    get_connectivity_results,
    get_node_results_polars,
    get_post_contingency_results,
    get_va_diff_results_polars,
    update_basename_polars,
)
from toop_engine_contingency_analysis.result_filter import apply_result_filter
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    CONTINGENCY_PROPAGATION,
    get_cgmes_distributed_slack_parameters,
    get_single_slack_parameters,
)
from toop_engine_grid_helpers.powsybl.polars.get_dataframe import (
    get_ca_branch_results,
    get_ca_bus_results,
    get_ca_three_windings_transformer_results,
)
from toop_engine_interfaces.loadflow_result_filter import LoadflowResultFilter
from toop_engine_interfaces.loadflow_result_helpers import convert_polars_loadflow_results_to_pandas
from toop_engine_interfaces.loadflow_result_helpers_polars import concatenate_loadflow_results_polars
from toop_engine_interfaces.loadflow_results import (
    ConvergenceStatus,
    LoadflowResults,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


@dataclass
class PowsyblBranchLimitCache:
    """Cached prepared branch limits for a specific monitored branch set and network limit snapshot.

    It contains the branch_limits dataframe as the core cached item, along with metadata needed to validate whether the
    cache can be reused safely.
    """

    chosen_limit: str
    """The limit name used when preparing the cached branch limits."""

    monitored_branches: tuple[str, ...]
    """The normalized monitored branch ids for which the cached limits were prepared."""

    current_limit_fingerprint: str
    """A fingerprint of the raw current-limit table used to detect stale cache entries."""

    branch_limits: pd.DataFrame
    """The prepared branch-limit dataframe ready for contingency-analysis postprocessing."""

    def matches(self, monitored_branches: Sequence[str], chosen_limit: str, current_limit_fingerprint: str) -> bool:
        """Check whether this cache can be reused for the current request.

        Parameters
        ----------
        monitored_branches : Sequence[str]
            The monitored branch ids needed for the current analysis.
        chosen_limit : str
            The selected operational limit name for the current analysis.
        current_limit_fingerprint : str
            The fingerprint of the current network's raw limit table.

        Returns
        -------
        bool
            True if the cache can be reused safely, otherwise False.
        """
        return (
            self.chosen_limit == chosen_limit
            and self.monitored_branches == normalize_monitored_branches_for_powsybl(monitored_branches)
            and self.current_limit_fingerprint == current_limit_fingerprint
        )


def build_branch_limit_cache(
    net: Network, monitored_branches: Sequence[str], chosen_limit: str = "permanent_limit"
) -> PowsyblBranchLimitCache:
    """Build a reusable cache for the expensive branch-limit preparation stage.

    Parameters
    ----------
    net : Network
        The active Powsybl network whose current limits are used for cache construction.
    monitored_branches : Sequence[str]
        The monitored branch ids for which limits should be prepared.
    chosen_limit : str, optional
        The limit name to select, by default "permanent_limit".

    Returns
    -------
    PowsyblBranchLimitCache
        A cache entry containing prepared limits and the metadata needed to validate reuse.
    """
    current_limits = get_current_branch_limits_for_powsybl(net)
    return PowsyblBranchLimitCache(
        chosen_limit=chosen_limit,
        monitored_branches=normalize_monitored_branches_for_powsybl(monitored_branches),
        current_limit_fingerprint=fingerprint_current_limits_for_powsybl(current_limits),
        branch_limits=prepare_branch_limits(
            branch_limits=current_limits,
            monitored_branches=list(monitored_branches),
            chosen_limit=chosen_limit,
        ),
    )


def _add_contingency_voltage_levels(
    analysis: pypowsybl.security.SecurityAnalysis, n_minus_1_definition: PowsyblNMinus1Definition
) -> None:
    """Monitor the voltage levels a contingency reports on top of the common ones.

    OpenLoadFlow reports a contingency with monitored elements of its own from those alone, without the ones monitored
    in every contingency, so each of these lists repeats the common elements. Contingencies with the same voltage
    levels share a list.
    """
    monitored = n_minus_1_definition.monitored_elements
    common = set(monitored["voltage_levels"])
    outage_ids = {contingency.id for contingency in n_minus_1_definition.contingencies if contingency.elements}
    by_voltage_levels: dict[tuple[str, ...], list[str]] = {}
    for contingency_id, voltage_levels in n_minus_1_definition.contingency_voltage_levels.items():
        extra = tuple(sorted(set(voltage_levels) - common))
        if extra and contingency_id in outage_ids:
            by_voltage_levels.setdefault(extra, []).append(contingency_id)
    for extra, contingency_ids in by_voltage_levels.items():
        analysis.add_monitored_elements(
            contingency_context_type=pypowsybl.security.ContingencyContextType.SPECIFIC,
            contingency_ids=contingency_ids,
            branch_ids=monitored["branches"],
            three_windings_transformer_ids=monitored["trafo3w"],
            voltage_level_ids=[*monitored["voltage_levels"], *extra],
        )


def run_powsybl_analysis(
    net: Network,
    n_minus_1_definition: PowsyblNMinus1Definition,
    security_analysis_params: pypowsybl.security.Parameters,
    method: Literal["ac", "dc"] = "ac",
) -> tuple[SecurityAnalysisResult, str | None]:
    """Run the powsybl security analysis for the given network and N-1 definition.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    security_analysis_params : pypowsybl.security.Parameters
        The security analysis parameters, used as they are, see get_monitored_security_analysis_params.
    method : Literal["ac", "dc"], optional
        The method to use for the contingency analysis. Either "ac" or "dc", by default "ac"

    Returns
    -------
    res: SecurityAnalysisResult
        The security analysis result from powsybl containing the monitored elements and the results of the contingencies.
    basecase_id : str | None
        The name of the basecase contingency, if it is included in the run. Otherwise None.
        Contingencies without elements other than the base case are not passed to powsybl; they have no impact.
    """
    # Run the actual loadflow computation
    analysis = pypowsybl.security.create_analysis()
    analysis.add_monitored_elements(
        branch_ids=n_minus_1_definition.monitored_elements["branches"],
        three_windings_transformer_ids=n_minus_1_definition.monitored_elements["trafo3w"],
        voltage_level_ids=n_minus_1_definition.monitored_elements["voltage_levels"],
    )
    _add_contingency_voltage_levels(analysis, n_minus_1_definition)
    basecase_id = None
    for contingency in n_minus_1_definition.contingencies:
        outages = contingency.elements
        if len(outages) > 1:
            analysis.add_multiple_elements_contingency(outages, contingency_id=contingency.id)
        elif len(outages) == 0:
            # Without a declared base case the empty contingency is the base case, as it always was
            if n_minus_1_definition.basecase_id in (None, contingency.id):
                basecase_id = contingency.id
        else:
            analysis.add_single_element_contingency(outages[0], contingency_id=contingency.id)

    if method == "ac":
        res = analysis.run_ac(net, security_analysis_params)
    else:
        res = analysis.run_dc(net, security_analysis_params)
    return res, basecase_id


def _outage_ids(pow_n1_definition: PowsyblNMinus1Definition, basecase_id: str | None) -> list[str]:
    """Every contingency of the definition except the base case."""
    return [
        contingency.id
        for contingency in pow_n1_definition.contingencies
        if not (contingency.is_basecase() and contingency.id == basecase_id)
    ]


def _security_analysis_raw(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    timestep: int,
    security_analysis_params: pypowsybl.security.Parameters,
    method: Literal["ac", "dc"],
    components: Optional[SynchronousComponents],
) -> tuple[RawSecurityResults, pd.DataFrame, list[str], str | None, pl.LazyFrame]:
    """Run the security analysis and complete its raw results.

    Contingencies without elements get the base case rows and status, and with ``fill_no_impact_contingencies`` of the
    definition so do the contingencies powsybl reports no result for (it leaves out contingencies that do not change
    the network). With *components*, the converged contingencies
    also get the base case rows of the synchronous components other than the main one, which OpenLoadFlow does not
    solve after a contingency. The last value returned is the connectivity result of the analysis.
    """
    ca_result, basecase_id = run_powsybl_analysis(
        net, pow_n1_definition, security_analysis_params=security_analysis_params, method=method
    )
    raw = conform_raw_results(
        RawSecurityResults(
            get_ca_bus_results(ca_result, lazy=True),
            get_ca_branch_results(ca_result, lazy=True),
            get_ca_three_windings_transformer_results(ca_result, lazy=True),
        )
    )
    post_contingency_results = ca_result.post_contingency_results
    pre_contingency_result = ca_result.pre_contingency_result
    outage_ids = _outage_ids(pow_n1_definition, basecase_id)
    convergence_df, failed_outages = get_convergence_result_df(
        post_contingency_results, pre_contingency_result, outage_ids, timestep, basecase_id
    )
    # A contingency without elements is not passed to powsybl and cannot change the grid. One with elements that powsybl
    # leaves out is only filled when asked for.
    without_elements = {contingency.id for contingency in pow_n1_definition.contingencies if not contingency.elements}
    no_impact = [
        outage
        for outage in outage_ids
        if outage not in post_contingency_results
        and (outage in without_elements or pow_n1_definition.fill_no_impact_contingencies)
    ]
    if no_impact:
        basecase_status = convert_powsybl_status(pre_contingency_result.status)
        convergence_df.loc[[(timestep, outage) for outage in no_impact], "status"] = basecase_status
        if basecase_status == ConvergenceStatus.CONVERGED.value:
            failed_outages = [outage for outage in failed_outages if outage not in no_impact]
            raw = copy_basecase_rows(raw, no_impact)
    if components is not None:
        solved = [outage for outage in outage_ids if outage not in failed_outages and outage not in no_impact]
        raw = copy_basecase_rows(raw, solved, components)
    connectivity_result = get_connectivity_results(get_post_contingency_results(list(post_contingency_results.values())))
    return raw, convergence_df, failed_outages, basecase_id, connectivity_result


# pylint: disable-next=too-many-positional-arguments
def _shape_results(
    raw: RawSecurityResults,
    convergence_df: pd.DataFrame,
    failed_outages: list[str],
    basecase_id: str | None,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"],
    connectivity_result: Optional[pl.LazyFrame] = None,
) -> LoadflowResultsPolars:
    """Turn raw security analysis frames into the loadflow result tables."""
    monitored_elements = pow_n1_definition.monitored_elements
    bus_results = raw.bus_results
    branch_results = raw.branch_results
    three_windings_transformer_results = raw.three_windings_transformer_results
    all_outage_ids = _outage_ids(pow_n1_definition, basecase_id)

    add_name_column(convergence_df, pow_n1_definition.contingency_name_mapping, index_level="contingency")
    convergence_df = pl.from_pandas(convergence_df, include_index=True, nan_to_null=False).lazy()

    branch_limit_polars = pl.from_pandas(pow_n1_definition.branch_limits, include_index=True, nan_to_null=False).lazy()

    branch_results_df = get_branch_results_polars(
        branch_results,
        three_windings_transformer_results,
        monitored_elements["branches"],
        monitored_elements["trafo3w"],
        failed_outages,
        timestep,
        branch_limit_polars,
    )
    node_results_df = get_node_results_polars(
        bus_results,
        monitored_elements["buses"],
        pl.from_pandas(pow_n1_definition.bus_map, include_index=True, nan_to_null=False).lazy(),
        pl.from_pandas(pow_n1_definition.voltage_levels, include_index=True, nan_to_null=False).lazy(),
        failed_outages,
        timestep,
        method,
    )
    regulating_elements_df = get_regulating_element_results(
        monitored_elements["buses"], timestep=timestep, basecase_name=basecase_id
    )
    regulating_elements_df = pl.from_pandas(regulating_elements_df, include_index=True, nan_to_null=False).lazy()
    va_diff_results_df = get_va_diff_results_polars(
        bus_results=bus_results,
        outages=all_outage_ids,
        va_diff_with_buses=pl.from_pandas(pow_n1_definition.blank_va_diff, include_index=True, nan_to_null=False).lazy(),
        bus_map=pl.from_pandas(pow_n1_definition.bus_map, include_index=True, nan_to_null=False).lazy(),
        timestep=timestep,
        bus_components=raw.bus_components,
    )
    branch_results_df = update_basename_polars(branch_results_df, basecase_id)
    branch_results_df = add_name_column_polars(
        branch_results_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    branch_results_df = add_name_column_polars(
        branch_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    branch_limits_df = get_branch_limits_polars(branch_limit_polars, timestep)

    node_results_df = update_basename_polars(node_results_df, basecase_id)
    node_results_df = add_name_column_polars(node_results_df, pow_n1_definition.element_name_mapping, index_level="element")
    node_results_df = add_name_column_polars(
        node_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    regulating_elements_df = update_basename_polars(regulating_elements_df, basecase_id)
    regulating_elements_df = add_name_column_polars(
        regulating_elements_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    regulating_elements_df = add_name_column_polars(
        regulating_elements_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    va_diff_results_df = update_basename_polars(va_diff_results_df, basecase_id)
    va_diff_results_df = add_name_column_polars(
        va_diff_results_df, pow_n1_definition.element_name_mapping, index_level="element"
    )
    va_diff_results_df = add_name_column_polars(
        va_diff_results_df, pow_n1_definition.contingency_name_mapping, index_level="contingency"
    )

    return LoadflowResultsPolars(
        job_id=job_id,
        branch_results=branch_results_df,
        branch_limits=branch_limits_df,
        node_results=node_results_df,
        regulating_element_results=regulating_elements_df,
        va_diff_results=va_diff_results_df,
        converged=convergence_df,
        connectivity_result=connectivity_result,
        warnings=[],
        lazy=True,
    )


def _run_contingency_analysis_polars(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    security_analysis_params: pypowsybl.security.Parameters,
    method: Literal["ac", "dc"] = "dc",
    components: Optional[SynchronousComponents] = None,
) -> LoadflowResultsPolars:
    """Compute the N-0 + N-1 power flow for the network.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for
    pow_n1_definition : PowsyblNMinus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep to use for the contingency analysis
    security_analysis_params : pypowsybl.security.Parameters
        The security analysis parameters, used as they are, see get_monitored_security_analysis_params. The
        parallelization is done by OpenLoadFlow via the provider parameter threadCount.
    method : Literal["ac", "dc"], optional
        Whether to compute the AC or DC power flow, by default "dc"
    components : Optional[SynchronousComponents], optional
        If given, contingencies get the base case values of the synchronous components other than the main one.

    Returns
    -------
    LoadflowResultsPolars
        The results of the loadflow computation. Invalid or otherwise failed results will be set to NaN.
    """
    raw, convergence_df, failed_outages, basecase_id, connectivity_result = _security_analysis_raw(
        net, pow_n1_definition, timestep, security_analysis_params, method, components
    )
    return _shape_results(
        raw, convergence_df, failed_outages, basecase_id, pow_n1_definition, job_id, timestep, method, connectivity_result
    )


def _run_loadflow_contingencies_polars(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    lf_params: pypowsybl.loadflow.Parameters,
    slack_overrides: Mapping[str, Sequence[str]],
) -> LoadflowResultsPolars:
    """Solve the (non-basecase) contingencies of the definition one by one, every island included."""
    monitored = pow_n1_definition.monitored_elements
    contingencies = {contingency.id: contingency.elements for contingency in pow_n1_definition.contingencies}
    raw, statuses = run_contingencies_as_loadflows(
        net,
        contingencies,
        lf_params,
        monitored_branches=monitored["branches"],
        monitored_trafo3w=monitored["trafo3w"],
        monitored_voltage_levels=monitored["voltage_levels"],
        slack_bus_overrides=slack_overrides,
        contingency_voltage_levels=pow_n1_definition.contingency_voltage_levels,
    )
    # A security analysis brings its pre-contingency rows along; here they are added, for the base case voltages
    extra_voltage_levels = {
        voltage_level
        for contingency_id in contingencies
        for voltage_level in pow_n1_definition.contingency_voltage_levels.get(contingency_id, ())
    }
    raw = concat_raw_results([basecase_bus_rows(net, lf_params, {*monitored["voltage_levels"], *extra_voltage_levels}), raw])
    outage_ids = list(contingencies)
    convergence_df = pd.DataFrame(
        {
            "status": [convert_powsybl_status(statuses[outage]) for outage in outage_ids],
            "iteration_count": float("nan"),
            "warnings": "",
            "contingency_name": "",
        },
        index=pd.MultiIndex.from_product([[timestep], outage_ids], names=["timestep", "contingency"]),
    )
    failed_outages = [outage for outage in outage_ids if statuses[outage] != pypowsybl.loadflow.ComponentStatus.CONVERGED]
    return _shape_results(raw, convergence_df, failed_outages, None, pow_n1_definition, job_id, timestep, "ac")


# pylint: disable-next=too-many-positional-arguments
def run_contingency_analysis_polars(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    security_analysis_params: pypowsybl.security.Parameters,
    method: Literal["ac", "dc"] = "dc",
    result_filter: Optional[LoadflowResultFilter] = None,
    batch_size: Optional[int] = None,
    components: Optional[SynchronousComponents] = None,
) -> LoadflowResultsPolars:
    """Compute N-0 + N-1 power flows, optionally processing outages in sequential batches.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the contingency analysis for.
    pow_n1_definition : PowsyblNMinus1Definition
        The translated N-1 definition to execute.
    job_id : str
        Identifier of the current job.
    timestep : int
        Timestep stored in the results.
    security_analysis_params : pypowsybl.security.Parameters
        The security analysis parameters used for each batch, see get_monitored_security_analysis_params.
    method : Literal["ac", "dc"], optional
        Whether to execute AC or DC power flows, by default "dc".
    result_filter : LoadflowResultFilter, optional
        Policy for dropping result rows that carry no decision value.
    batch_size : int, optional
        Maximum number of outage contingencies per sequential Powsybl analysis. The N-0 basecase, if present, is run
        with the first batch and does not count towards this limit. If None, all contingencies are processed together.
    components : Optional[SynchronousComponents], optional
        If given, contingencies get the base case values of the synchronous components other than the main one.

    Returns
    -------
    LoadflowResultsPolars
        Combined results from every sequential batch.

    Raises
    ------
    ValueError
        If batch_size is not positive.
    """
    basecase_contingencies = [
        contingency
        for contingency in pow_n1_definition.contingencies
        if contingency.is_basecase() and pow_n1_definition.basecase_id in (None, contingency.id)
    ]
    basecase_ids = {contingency.id for contingency in basecase_contingencies}
    if batch_size is None:
        contingency_batches = [pow_n1_definition.contingencies]
    else:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outage_contingencies = [
            contingency for contingency in pow_n1_definition.contingencies if contingency.id not in basecase_ids
        ]
        outage_batches = [
            outage_contingencies[start : start + batch_size] for start in range(0, len(outage_contingencies), batch_size)
        ]
        contingency_batches = (
            [[*basecase_contingencies, *outage_batches[0]], *outage_batches[1:]]
            if outage_batches
            else [basecase_contingencies]
        )
    batch_results = [
        _run_contingency_analysis_polars(
            net,
            pow_n1_definition.model_copy(update={"contingencies": contingencies}),
            job_id,
            timestep,
            security_analysis_params,
            method,
            components,
        )
        for contingencies in contingency_batches
    ]
    lf_results = batch_results[0] if len(batch_results) == 1 else concatenate_loadflow_results_polars(batch_results)
    if result_filter is None:
        return lf_results
    return apply_result_filter(lf_results, result_filter, basecase_contingencies[0].id if basecase_contingencies else None)


def get_monitored_security_analysis_params(
    security_analysis_params: pypowsybl.security.Parameters, method: Literal["ac", "dc"]
) -> pypowsybl.security.Parameters:
    """Get the security analysis parameters as the monitored element runner runs them.

    A DC loadflow does not support a distributed slack, so it is switched off for DC. Everything else, including the
    contingency propagation, is used as given.

    Parameters
    ----------
    security_analysis_params : pypowsybl.security.Parameters
        The security analysis parameters of the caller. They are not modified.
    method : Literal["ac", "dc"]
        Whether AC or DC power flows are run.

    Returns
    -------
    pypowsybl.security.Parameters
        A copy, for DC without a distributed slack.
    """
    security_analysis_params = deepcopy(security_analysis_params)
    if method == "dc":
        security_analysis_params.load_flow_parameters.distributed_slack = False
    return security_analysis_params


def _split_contingencies(
    pow_n1_definition: PowsyblNMinus1Definition,
    loadflow_ids: set[str],
    slack_overrides: Mapping[str, Sequence[str]],
) -> tuple[PowsyblNMinus1Definition, dict[tuple[str, ...], PowsyblNMinus1Definition], PowsyblNMinus1Definition]:
    """Split a definition into the main security analysis, one per distinct slack override, and the load flow ones."""
    main, by_slack, loadflow = [], {}, []
    for contingency in pow_n1_definition.contingencies:
        if contingency.id in loadflow_ids and not contingency.is_basecase():
            loadflow.append(contingency)
        elif contingency.id in slack_overrides and not contingency.is_basecase():
            by_slack.setdefault(tuple(slack_overrides[contingency.id]), []).append(contingency)
        else:
            main.append(contingency)

    def subset(contingencies: list) -> PowsyblNMinus1Definition:
        return pow_n1_definition.model_copy(update={"contingencies": contingencies})

    return subset(main), {key: subset(value) for key, value in by_slack.items()}, subset(loadflow)


# pylint: disable-next=too-many-positional-arguments
def _run_with_overrides(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"],
    security_analysis_params: pypowsybl.security.Parameters,
    batch_size: Optional[int],
    components: Optional[SynchronousComponents],
    loadflow_ids: set[str],
    slack_overrides: Mapping[str, Sequence[str]],
) -> LoadflowResultsPolars:
    """Run the main security analysis, one per slack override, and the per-contingency load flows; concatenate."""
    if method != "ac":
        raise ValueError("Slack overrides and per-contingency load flows need method='ac'")
    main, by_slack, loadflow = _split_contingencies(pow_n1_definition, loadflow_ids, slack_overrides)
    common = {
        "net": net,
        "job_id": job_id,
        "timestep": timestep,
        "method": method,
        "batch_size": batch_size,
        "components": components,
    }
    lf_params = security_analysis_params.load_flow_parameters
    parts = [
        run_contingency_analysis_polars(pow_n1_definition=main, security_analysis_params=security_analysis_params, **common)
    ]
    for slack_ids, definition in by_slack.items():
        slack_params = deepcopy(security_analysis_params)
        slack_params.load_flow_parameters = with_slack_buses(lf_params, resolve_slack_buses(net, slack_ids))
        parts.append(
            run_contingency_analysis_polars(pow_n1_definition=definition, security_analysis_params=slack_params, **common)
        )
    if loadflow.contingencies:
        parts.append(_run_loadflow_contingencies_polars(net, loadflow, job_id, timestep, lf_params, slack_overrides))
    return concatenate_loadflow_results_polars(parts)


def _non_main_components(
    net: Network,
    pow_n1_definition: PowsyblNMinus1Definition,
    lf_params: pypowsybl.loadflow.Parameters,
    method: Literal["ac", "dc"],
) -> tuple[SynchronousComponents, set[str]]:
    """Return the synchronous components of the network and the contingencies with an element outside the main one."""
    if method != "ac":
        raise ValueError("fill_non_main_components needs method='ac'")
    components = get_synchronous_components(net, lf_params)
    contingencies = {contingency.id: contingency.elements for contingency in pow_n1_definition.contingencies}
    return components, contingencies_touching_non_main_components(net, contingencies, components)


# pylint: disable-next=too-many-positional-arguments,too-many-locals
def run_contingency_analysis_powsybl(  # noqa: PLR0913
    net: Network,
    n_minus_1_definition: Nminus1Definition,
    job_id: str,
    timestep: int,
    method: Literal["ac", "dc"] = "ac",
    *,
    polars: bool = False,
    branch_limit_cache: Optional[PowsyblBranchLimitCache] = None,
    result_filter: Optional[LoadflowResultFilter] = None,
    batch_size: Optional[int] = None,
    security_analysis_params: Optional[pypowsybl.security.Parameters] = None,
    slack_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    loadflow_contingency_ids: Optional[Collection[str]] = None,
    fill_non_main_components: bool = False,
    fill_no_impact_contingencies: bool = False,
    monitored_voltage_levels: Optional[Collection[str]] = None,
    contingency_voltage_levels: Optional[Mapping[str, Collection[str]]] = None,
) -> Union[LoadflowResults, LoadflowResultsPolars]:
    """Compute the Contingency Analysis for the network.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the Contingency Analysis for. It is analysed as it is given: the DC analysis
        runs without a distributed slack, so to keep the N-0 flows of a distributed slack, balance the grid first
        with toop_engine_grid_helpers.powsybl.powsybl_helpers.balance_grid.
    n_minus_1_definition : Nminus1Definition
        The N-1 definition to use for the contingency analysis. Contains outages and monitored elements
    job_id : str
        The job id of the current job
    timestep : int
        The timestep to use for the contingency analysis
    method : Literal["ac", "dc"], optional
        The method to use for the contingency analysis. Either "ac" or "dc", by default "dc"
    polars: bool
        Whether to use polars for the dataframe operations.
    branch_limit_cache : Optional[PowsyblBranchLimitCache]
        Optional cache for the expensive prepared branch limits. If stale or None, it will be ignored and recomputed.
        If you want to run many contingency analyses with the same network, use this and precompute the cache with
        build_branch_limit_cache() to save time.
    result_filter : Optional[LoadflowResultFilter]
        Policy for dropping result rows that carry no decision value. If None, every row is kept.
    batch_size : int, optional
        Maximum number of outage contingencies per sequential Powsybl analysis. If None, all contingencies are processed
        together.
    security_analysis_params : Optional[pypowsybl.security.Parameters]
        The security analysis parameters, including the loadflow parameters and the provider parameters such as
        threadCount and contingencyPropagation. They are not modified and used as they are, except that the distributed
        slack is switched off for DC, see get_monitored_security_analysis_params. The contingencies are run as they
        are defined: with contingencyPropagation "false", they have to be propagated already, see
        propagate_busbar_contingencies. If None, get_cgmes_distributed_slack_parameters (AC) or
        get_single_slack_parameters (DC) is used with a single thread and the contingency propagation of powsybl.
    slack_overrides : Optional[Mapping[str, Sequence[str]]]
        Contingency id to the slack to use instead of the one of the load flow parameters, as injection ids (their
        bus) or bus ids. AC only. Contingencies with the same override share a security analysis.
    loadflow_contingency_ids : Optional[Collection[str]]
        Contingencies to solve with their own AC load flow over every island instead of in the security analysis
        (which solves only the main synchronous component). With a slack override, only the islands holding one of the
        given slacks are solved.
    fill_non_main_components : bool
        Report the synchronous components other than the main one too: contingencies outside them get their base case
        values, contingencies inside them are solved with their own load flow. AC only.
    fill_no_impact_contingencies : bool
        Give contingencies powsybl reports no result for the base case values and status. Powsybl leaves out
        contingencies that do not change the network, e.g. the outage of an element that is already disconnected. By
        default they have no result rows and the status NO_CALCULATION. A contingency without elements always gets
        the base case values.
    monitored_voltage_levels : Optional[Collection[str]]
        The voltage levels whose buses are reported in every contingency. If None, they are those of the monitored
        buses and switches and of both ends of the monitored and outaged branches. A monitored bus outside them has
        no result. Bus results are the largest part of the result, so give them when fewer are needed.
    contingency_voltage_levels : Optional[Mapping[str, Collection[str]]]
        Contingency id to voltage levels reported in that contingency only, on top of the ones above.

    Returns
    -------
    Union[LoadflowResults, LoadflowResultsPolars]
        The results of the loadflow computation.
    """
    if security_analysis_params is None:
        security_analysis_params = pypowsybl.security.Parameters(
            load_flow_parameters=get_cgmes_distributed_slack_parameters()
            if method == "ac"
            else get_single_slack_parameters(),
            provider_parameters={"threadCount": "1", CONTINGENCY_PROPAGATION: "true"},
        )
    security_analysis_params = get_monitored_security_analysis_params(security_analysis_params, method)
    pow_n1_definition = translate_nminus1_for_powsybl(
        n_minus_1_definition,
        net,
        branch_limit_cache=branch_limit_cache,
    )
    pow_n1_definition.fill_no_impact_contingencies = fill_no_impact_contingencies
    if monitored_voltage_levels is not None:
        pow_n1_definition.monitored_elements["voltage_levels"] = sorted(set(monitored_voltage_levels))
    if contingency_voltage_levels:
        pow_n1_definition.contingency_voltage_levels = {
            contingency_id: sorted(set(voltage_levels))
            for contingency_id, voltage_levels in contingency_voltage_levels.items()
        }

    slack_overrides = dict(slack_overrides or {})
    loadflow_ids = set(loadflow_contingency_ids or ())
    components = None
    if fill_non_main_components:
        components, touching = _non_main_components(
            net, pow_n1_definition, security_analysis_params.load_flow_parameters, method
        )
        loadflow_ids |= touching

    if not slack_overrides and not loadflow_ids:
        lf_result = run_contingency_analysis_polars(
            net=net,
            pow_n1_definition=pow_n1_definition,
            job_id=job_id,
            timestep=timestep,
            method=method,
            security_analysis_params=security_analysis_params,
            result_filter=result_filter,
            batch_size=batch_size,
            components=components,
        )
    else:
        lf_result = _run_with_overrides(
            net,
            pow_n1_definition,
            job_id,
            timestep,
            method,
            security_analysis_params,
            batch_size,
            components,
            loadflow_ids,
            slack_overrides,
        )
        basecase = n_minus_1_definition.base_case
        if result_filter is not None:
            lf_result = apply_result_filter(lf_result, result_filter, basecase.id if basecase is not None else None)
    if not polars:
        lf_result = convert_polars_loadflow_results_to_pandas(lf_result)
    missing_element_warnings = [
        f"Element with id {element.id} not found in the network." for element in pow_n1_definition.missing_elements
    ]
    missing_contingency_warnings = [
        f"Contingency with id {contingency.id} contains elements that are not found in the network."
        for contingency in pow_n1_definition.missing_contingencies
    ]
    lf_result.warnings = [*missing_element_warnings, *missing_contingency_warnings, *lf_result.warnings]
    return lf_result
