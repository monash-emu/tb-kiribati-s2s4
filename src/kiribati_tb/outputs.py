"""Derived outputs (``tbh/outputs.py``).

:func:`request_model_outputs` asks for the same outputs, by the same names, in the same order
as the original, including the ``save_results=False`` intermediates (``keep=False`` here). Each
summer2 request has a one-call summer4 counterpart on ``OutputSet``:

=====================================================  ==========================================
summer2                                                summer4
=====================================================  ==========================================
``request_output_for_compartments(n, comps, strata)``  ``outputs.compartments(n, where)``
``request_output_for_flow(n, flow, source_strata)``    ``outputs.flow(n, flow, source=where)``
``request_aggregate_output(n, sources)``               ``outputs.aggregate(n, sources)``
``request_function_output(n, DerivedOutput(a) / ..)``  ``outputs[n] = outputs.ref(a) / ...``
``request_cumulative_output(n, src, start_time)``      ``outputs.cumsum`` / ``cumulative_integral``
``request_computed_value_output(n)``                   ``outputs[n] = <rate expression>``
``save_results=False``                                 ``keep=False``
a ``for`` loop over age and reachability strata        ``by=[AGE, REACH, (AGE, REACH)]``
=====================================================  ==========================================

``by=`` names each stratum ``{name}X{property}_{trait}``. The properties are called ``age``
and ``reach``, so ``by=(AGE, REACH)`` gives the original's ``prev_incipientXage_15Xreach_
reachable``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from summer4 import OutputSet, Param

from kiribati_tb.variants import Variant

REACHABLE = "reachable"
UNREACHABLE = "unreachable"
CUMULATIVE_START = 2026.0


def request_model_outputs(
    compartments: Sequence[str],
    active_compartments: Sequence[str],
    latent_compartments: Sequence[str],
    nat_death_flows: Mapping[str, tuple[str, Any]],
    tb_death_flows: Mapping[str, tuple[str, Any]],
    screening_flows: Sequence[str],
    computed_values: Mapping[str, Any],
    variant: Variant,
    flow_sources: Mapping[str, tuple[str, Any, Any]] | None = None,
) -> OutputSet:
    """Every derived output of the original model, by the original's names.

    Args:
        compartments, active_compartments, latent_compartments: TB state names.
        nat_death_flows, tb_death_flows: ``{output name: (flow, source selector)}``.
        screening_flows: Screening flow names.
        computed_values: ``{output name: rate expression}`` to save during the solve.
        variant: Sets the flow-output convention and how cumulative outputs are formed.
        flow_sources: For a model that merges the original's flows
            (:mod:`kiribati_tb.model_summer4`): original flow name -> ``(flow, source, dest)``.
    """
    from kiribati_tb.model import AGE, REACH, STATE  # the model module imports this one

    age_strata = AGE.traits
    reach_strata = REACH.traits
    outputs = OutputSet(flow_values=variant.flow_values)
    cumulative = outputs.cumsum if variant.cumulative == "cumsum" else outputs.cumulative_integral
    merged = dict(flow_sources or {})

    def flow(name: str, original: str, source: Any = None, keep: bool = True) -> None:
        """``outputs.flow`` on an original flow name, or on the flow that now carries it."""
        actual, merged_source, dest = merged.get(original, (original, None, None))
        if merged_source is not None:
            source = merged_source if source is None else merged_source & source
        outputs.flow(name, actual, source=source, dest=dest, keep=keep)

    reachable = REACH[REACHABLE]
    age_aggregates = {
        "3_9": ["3", "5"],
        "15+": [age for age in age_strata if int(age) >= 15],
        "18+": [age for age in age_strata if int(age) >= 18],
    }

    # Population size, by age and by reachability; by age among the reachable is intermediate.
    outputs.compartments("population", by=[AGE, REACH])
    for age in age_strata:
        outputs.compartments(
            f"populationXage_{age}Xreach_{REACHABLE}", AGE[age] & reachable, keep=False
        )
    for label, ages in age_aggregates.items():
        outputs.aggregate(f"populationXage_{label}", [f"populationXage_{a}" for a in ages])
        outputs.aggregate(
            f"populationXage_{label}Xreach_{REACHABLE}",
            [f"populationXage_{a}Xreach_{REACHABLE}" for a in ages],
            keep=False,
        )

    # Births (only those adding to the population, not deaths reintroduced as newborns)
    for reach in reach_strata:
        flow(f"births_{reach}", f"births_{reach}")
    outputs.aggregate("births", [f"births_{reach}" for reach in reach_strata])

    # TB incidence (and cumulative)
    for inf_cat in ["lowinf", "inf"]:
        flow(f"tb_incidence_{inf_cat}", f"progression_{inf_cat}")
        for reach in reach_strata:
            flow(
                f"tb_incidence_{inf_cat}Xreach_{reach}",
                f"progression_{inf_cat}",
                source=REACH[reach],
                keep=False,
            )
    for reach in reach_strata:
        outputs.aggregate(
            f"tb_incidenceXreach_{reach}",
            [f"tb_incidence_{inf_cat}Xreach_{reach}" for inf_cat in ["lowinf", "inf"]],
        )
    outputs.aggregate("tb_incidence", [f"tb_incidenceXreach_{reach}" for reach in reach_strata])
    request_per_capita_output(outputs, "tb_incidence", per=100000.0)
    outputs[f"prop_tb_incidenceXreach_{UNREACHABLE}"] = outputs.ref(
        f"tb_incidenceXreach_{UNREACHABLE}"
    ) / outputs.ref("tb_incidence")
    cumulative("cum_tb_incidence", "tb_incidence", start=CUMULATIVE_START)

    # True absolute prevalence (compartment sizes), by age, reachability and both
    for comp in compartments:
        outputs.compartments(f"prev_{comp}", STATE[comp], by=[AGE, REACH, (AGE, REACH)], keep=False)
    # True per-capita prevalence for TB and TBI
    for state, comp_list, per in zip(
        ["tbi", "tb"], [latent_compartments, active_compartments], [100.0, 100000.0]
    ):
        outputs.aggregate(f"{state}_prevalence", [f"prev_{comp}" for comp in comp_list])
        request_per_capita_output(outputs, f"{state}_prevalence", per=per)

    # Measured TST positives (compartment-specific TST sensitivity), among the reachable
    for comp in compartments:
        if comp in latent_compartments:
            tst_sensitivity: Any = Param(f"prev_se_{comp}_tst")
        elif comp in active_compartments or comp == "treatment":
            tst_sensitivity = 1.0
        elif comp == "recovered":
            tst_sensitivity = Param("prev_se_cleared_tst")
        else:  # mtb_naive
            tst_sensitivity = 0.0
        for age in age_strata:
            outputs.set(
                f"tst_pos_{comp}Xage_{age}Xreach_{REACHABLE}",
                outputs.ref(f"prev_{comp}Xage_{age}Xreach_{REACHABLE}") * tst_sensitivity,
                keep=False,
            )
        for label, ages in age_aggregates.items():
            outputs.aggregate(
                f"tst_pos_{comp}Xage_{label}Xreach_{REACHABLE}",
                [f"tst_pos_{comp}Xage_{a}Xreach_{REACHABLE}" for a in ages],
            )
    # Per-capita TST positivity for each age group and aggregate
    for age in [*age_strata, *age_aggregates]:
        name = f"tst_posXage_{age}Xreach_{REACHABLE}"
        outputs.aggregate(
            name, [f"tst_pos_{comp}Xage_{age}Xreach_{REACHABLE}" for comp in compartments]
        )
        request_per_capita_output(
            outputs, name, per=100.0, denominator_output=f"populationXage_{age}Xreach_{REACHABLE}"
        )
    name = f"tst_posXreach_{REACHABLE}"
    outputs.aggregate(name, [f"tst_posXage_{age}Xreach_{REACHABLE}" for age in age_strata])
    request_per_capita_output(
        outputs, name, per=100.0, denominator_output=f"populationXreach_{REACHABLE}"
    )

    # Measured PEARL (Xpert) and CXR positivity (test-specific sensitivity)
    for comp in active_compartments:
        for age in age_strata:
            for test in ["pearl", "cxr"]:
                outputs.set(
                    f"{test}_prev_{comp}Xage_{age}Xreach_{REACHABLE}",
                    outputs.ref(f"prev_{comp}Xage_{age}Xreach_{REACHABLE}")
                    * Param(f"prev_se_{comp}_{test}"),
                    keep=False,
                )
    for test in ["pearl", "cxr"]:
        for age in age_strata:
            name = f"{test}_posXage_{age}Xreach_{REACHABLE}"
            outputs.aggregate(
                name,
                [f"{test}_prev_{comp}Xage_{age}Xreach_{REACHABLE}" for comp in active_compartments],
            )
            request_per_capita_output(
                outputs,
                name,
                per=100000.0,
                denominator_output=f"populationXage_{age}Xreach_{REACHABLE}",
            )
        name = f"{test}_posXreach_{REACHABLE}"
        outputs.aggregate(name, [f"{test}_posXage_{age}Xreach_{REACHABLE}" for age in age_strata])
        request_per_capita_output(
            outputs, name, per=100000.0, denominator_output=f"populationXreach_{REACHABLE}"
        )

    # Prevalence of viable TB infection ("incipient" and "contained")
    outputs.aggregate("viable_tbi_prevalence", ["prev_incipient", "prev_contained"])
    request_per_capita_output(outputs, "viable_tbi_prevalence", per=100.0)

    # Percentage subclinical (compare with Frascella et al., CID 2020)
    subclinical = STATE.isin([c for c in active_compartments if c.startswith("subclin_")])
    outputs.compartments(
        f"subclin_tb_prevalenceXreach_{REACHABLE}", subclinical & reachable, keep=False
    )
    for reach in reach_strata:
        outputs.aggregate(
            f"tb_prevalenceXreach_{reach}",
            [f"prev_{comp}Xreach_{reach}" for comp in active_compartments],
        )
        request_per_capita_output(
            outputs,
            f"tb_prevalenceXreach_{reach}",
            per=100000.0,
            denominator_output=f"populationXreach_{reach}",
        )
    outputs[f"perc_prev_subclinicalXreach_{REACHABLE}"] = (
        100.0
        * outputs.ref(f"subclin_tb_prevalenceXreach_{REACHABLE}")
        / outputs.ref(f"tb_prevalenceXreach_{REACHABLE}")
    )
    # Percentage infectious
    infectious = STATE.isin([c for c in active_compartments if c.endswith("_inf")])
    outputs.compartments(
        f"infectious_tb_prevalenceXreach_{REACHABLE}", infectious & reachable, keep=False
    )
    outputs[f"perc_prev_infectiousXreach_{REACHABLE}"] = (
        100.0
        * outputs.ref(f"infectious_tb_prevalenceXreach_{REACHABLE}")
        / outputs.ref(f"tb_prevalenceXreach_{REACHABLE}")
    )

    # TB notifications
    for active_comp in active_compartments:
        flow(f"notifications_{active_comp}", f"tb_detection_{active_comp}", keep=False)
    outputs.aggregate(
        "notifications", [f"notifications_{active_comp}" for active_comp in active_compartments]
    )
    outputs["perc_notifications_clin"] = (
        100.0
        * (outputs.ref("notifications_clin_lowinf") + outputs.ref("notifications_clin_inf"))
        / outputs.ref("notifications")
    )

    # Screening
    for scr_flow in screening_flows:
        flow(scr_flow, scr_flow, keep=False)
    if screening_flows:
        outputs.aggregate("screening", screening_flows)
    else:  # summer2 aggregates an empty list to zeros; summer4 asks for it explicitly
        outputs["screening"] = 0.0 * outputs.ref("births")

    # Mortality
    for name, (death_flow, source) in nat_death_flows.items():
        outputs.flow(name, death_flow, source=source, keep=False)
    outputs.aggregate("nat_mortality", list(nat_death_flows))
    for name, (death_flow, source) in tb_death_flows.items():
        outputs.flow(name, death_flow, source=source, keep=False)
    outputs.aggregate("tb_mortality", list(tb_death_flows))
    request_per_capita_output(outputs, "tb_mortality", per=100000.0)
    cumulative("cum_tb_mortality", "tb_mortality", start=CUMULATIVE_START)

    # Computed values: passive detection rates and the distance to the conmat matrix. Each is
    # the rate expression the model uses, saved during the solve.
    for name, expression in computed_values.items():
        outputs[name] = expression

    return outputs


def request_per_capita_output(
    outputs: OutputSet, output: str, per: float = 100.0, denominator_output: str = "population"
) -> None:
    """``{output}_{perc|per100k}`` = ``per * output / denominator_output``."""
    suffix = {100.0: "perc", 100000.0: "per100k"}.get(per, f"per{per}")
    outputs[f"{output}_{suffix}"] = per * outputs.ref(output) / outputs.ref(denominator_output)
