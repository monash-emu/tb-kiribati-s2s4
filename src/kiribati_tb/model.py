"""The Kiribati TB model in summer4 (``tbh/model.py``).

The functions are the original's, in the original's order, and do the same things:

1. :func:`get_natural_tb_model` builds the unstratified model: ten TB states, infection and
   reinfection, latency, progression, clinical and infectiousness transitions, self-recovery.
2. :func:`add_detection_and_treatment` adds passive detection, screening and treatment.
3. :func:`stratify_model_by_age` splits every compartment by age, adds ageing and sets the
   age-specific flow adjustments.
4. :func:`stratify_model_by_reachability` splits by reachability and sets its adjustments.
5. :func:`add_births_and_deaths` adds deaths (recycled to the youngest age) and births.
6. :mod:`kiribati_tb.outputs` requests the derived outputs.

What changes is the vocabulary: a ``Property`` and its traits for summer2's strata, selectors
(``STATE["clin_inf"] & AGE["0"]``) for ``source_strata``/``dest_strata``, ``Multiply`` for
``set_flow_adjustments``, and rate expressions (``Param``, ``Time()``, arithmetic, ``Data``
interpolation, ``defer``) for summer2's ``Parameter``, ``Time`` and ``Function``.

Two structural simplifications summer4 allows (results are identical):

- Deaths are one flow per cause with a per-age rate, not one flow per (cause, compartment,
  age); the original's per-(compartment, age) output names are selections of that flow.
- The force of infection is declared whole, with its age mixing and infectiousness, where the
  infection flows are added (summer2 attaches those to the age stratification).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from summer4 import (
    REMAINDER,
    CompiledModel,
    EntryFlow,
    FlowModel,
    InitialPopulation,
    Multiply,
    OutputSet,
    Param,
    Property,
    PropertyMap,
    Split,
    Time,
    TraitChain,
    TransitionFlow,
    defer,
    field_refs,
    tanh,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix

from kiribati_tb.age_mixing import canberra_distance, get_model_ready_age_mixing_matrix
from kiribati_tb.age_mixing import read_conmat_matrix
from kiribati_tb.demographic_tools import AGE_GROUPS, load_demography
from kiribati_tb.interventions import ScreeningProgram
from kiribati_tb.outputs import request_model_outputs
from kiribati_tb.variants import PARITY, Variant

COMPARTMENTS = (
    "mtb_naive",  # historically called "susceptible"
    # Early TB infection states
    "incipient",
    "contained",
    "cleared",
    # TB disease states
    "subclin_lowinf",
    "clin_lowinf",
    "subclin_inf",
    "clin_inf",
    # Treatment and recovered
    "treatment",
    "recovered",
)
LATENT_COMPS = ["incipient", "contained", "cleared"]
ACTIVE_COMPS = ["subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"]
INFECTIOUS_COMPARTMENTS = ACTIVE_COMPS
REACH_STRATA = ("reachable", "unreachable")

# summer2's compartments and strata become properties: the state, then the two stratifications.
STATE = Property("state", COMPARTMENTS)
AGE = Property("age", AGE_GROUPS)
REACH = Property("reach", REACH_STRATA)
CHILDREN = AGE.isin([a for a in AGE_GROUPS if int(a) < 15])
NEWBORN = STATE["mtb_naive"] & AGE[AGE_GROUPS[0]]

# summer2's ``get_sigmoidal_interpolation_function`` defaults to curvature 16. summer4's
# ``sharpness`` is the same parameter but defaults to 1 (near-linear), so pass it.
SUMMER2_SIGMOID_CURVATURE = 16.0


class TBModel(NamedTuple):
    """What :func:`get_tb_model` returns: the compiled model and its derived outputs.

    summer2 keeps the output requests on the model; summer4 keeps them in an ``OutputSet`` that
    is passed to ``run(outputs=...)``, so one compiled model can be run with different sets.
    """

    model: CompiledModel
    outputs: OutputSet
    breakpoints: tuple[float, ...] = ()
    """Times where a rate has a kink (the screening ramps); the improved variant saves and
    steps on them."""


def get_tb_model(
    model_config: dict,
    tv_params: dict[str, pd.Series],
    screening_programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...] = (),
    variant: Variant = PARITY,
) -> TBModel:
    """Build the model and its outputs (``tbh.model.get_tb_model``).

    Args:
        model_config: ``DEFAULT_MODEL_CONFIG`` from :mod:`kiribati_tb.runner_tools`, or a copy
            with ``heterogeneous_mixing=False`` for the homogeneous sensitivity analysis.
        tv_params: Time-variant parameters (``tx_success_pct``) from ``parameters.xlsx``.
        screening_programs: Screening programs of a scenario; none for the baseline.
        variant: ``PARITY`` (summer2's flow-output convention) or ``IMPROVED``; it only
            changes the outputs here (the solver is chosen when running).
    """
    if tuple(model_config["age_groups"]) != AGE_GROUPS:
        raise ValueError(f"This port is built for age groups {AGE_GROUPS}")
    demog = load_demography(model_config["iso3"], AGE_GROUPS, model_config["pop_scaling"])
    agg_pop_data = demog.total
    bckd_death_funcs = demog.death_rate_table(AGE).interp(
        "sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE
    )
    tsr = tv_params["tx_success_pct"]
    time_variant_tsr = Data(tsr.index.to_numpy(float), tsr.to_numpy(float) / 100.0).interp("linear")
    neg_tx_outcome_funcs = get_neg_tx_outcome_funcs(bckd_death_funcs, time_variant_tsr)

    hetero = model_config.get("heterogeneous_mixing", True)
    age_mixing_matrix = get_model_ready_age_mixing_matrix() if hetero else None

    # Model building
    model, infection_flows = get_natural_tb_model(model_config, agg_pop_data, age_mixing_matrix)
    screening_flows, tv_detection_rate = add_detection_and_treatment(
        model, time_variant_tsr, screening_programs
    )
    stratify_model_by_age(model, neg_tx_outcome_funcs, screening_programs)
    stratify_model_by_reachability(model, infection_flows, screening_flows)
    nat_death_flows, tb_death_flows = add_births_and_deaths(
        model, agg_pop_data, bckd_death_funcs, neg_tx_outcome_funcs
    )

    # Computed values are rate expressions; the outputs save them (summer2's
    # add_computed_value_func + request_computed_value_output).
    computed_values = {
        "passive_detection_rate_clin": tv_detection_rate,
        "passive_detection_rate_subclin": Param("rel_detection_subclin") * tv_detection_rate,
    }
    if age_mixing_matrix is not None:
        conmat_matrix = read_conmat_matrix(model_config["iso3"], AGE_GROUPS)
        computed_values["mixing_matrix_distance"] = defer(canberra_distance)(
            age_mixing_matrix, conmat_matrix
        )

    outputs = request_model_outputs(
        COMPARTMENTS,
        ACTIVE_COMPS,
        LATENT_COMPS,
        nat_death_flows,
        tb_death_flows,
        screening_flows,
        computed_values,
        variant,
    )
    breakpoints = tuple(
        sorted(
            {
                float(t)
                for prog in screening_programs
                for t in (
                    prog.start_time - 0.01,
                    prog.start_time,
                    prog.end_time - 0.01,
                    prog.end_time,
                )
            }
        )
    )
    return TBModel(model.compile(), outputs, breakpoints)


def get_natural_tb_model(
    model_config: dict, pop_size: pd.Series, age_mixing_matrix: Any = None
) -> tuple[FlowModel, list[str]]:
    """The unstratified natural-history model; returns it and the infection flow names."""
    model = FlowModel(PropertyMap.from_property(STATE))

    seed = model_config["seed"]
    # The reachability split is declared here; summer4 applies it once the map has ``reach``.
    # Age is split evenly, as summer2's AgeStratification does by default.
    model.set_initial_population(
        InitialPopulation(
            {STATE["mtb_naive"]: pop_size.iloc[0] - seed, STATE["clin_inf"]: seed},
            splits=(
                Split(REACH, {"reachable": Param("reachable_pop_frac"), "unreachable": REMAINDER}),
            ),
        )
    )

    infection_pop_scale = Param("infection_pop_scale")
    # Scale the raw transmission rate by population size in 2020 (arbitrary choice), so that
    # "raw_transmission_rate" keeps the same order of magnitude across transmission modes.
    rescaled_transmission_rate = Param("raw_transmission_rate") * pop_size.loc[2020] ** (
        infection_pop_scale
    )
    foi = get_force_of_infection(rescaled_transmission_rate, infection_pop_scale, age_mixing_matrix)

    # Transmission flows (including reinfection)
    infection_flows = []
    for susceptible_comp in ["mtb_naive", "contained", "cleared", "recovered"]:
        if susceptible_comp == "recovered":  # common rel. susceptibility after clearance/recovery
            rel_susceptibility = Param("rel_sus_cleared")
        else:
            rel_susceptibility = (
                1.0 if susceptible_comp == "mtb_naive" else Param(f"rel_sus_{susceptible_comp}")
            )
        flowname = f"infection_from_{susceptible_comp}"
        model.add_flow(
            TransitionFlow(
                flowname,
                STATE[susceptible_comp],
                STATE["incipient"],
                foi,
                adjust=(Multiply(rel_susceptibility),),
            )
        )
        infection_flows.append(flowname)

    # Early TB infection dynamics
    model.add_flow(TransitionFlow("containment", STATE["incipient"], STATE["contained"], 1.0))
    model.add_flow(
        TransitionFlow("clearance", STATE["contained"], STATE["cleared"], Param("clearance_rate"))
    )
    model.add_flow(
        TransitionFlow("breakdown", STATE["contained"], STATE["incipient"], Param("breakdown_rate"))
    )
    # Progression splits between low and high infectious categories (all subclinical); the
    # rates are later adjusted by age.
    prop_inf = Param("progression_prop_infectious")
    model.add_flow(
        TransitionFlow(
            "progression_lowinf", STATE["incipient"], STATE["subclin_lowinf"], 1.0 - prop_inf
        )
    )
    model.add_flow(
        TransitionFlow("progression_inf", STATE["incipient"], STATE["subclin_inf"], prop_inf)
    )

    # Active TB dynamics: clinical progression and regression
    for infectious_status in ["inf", "lowinf"]:
        model.add_flow(
            TransitionFlow(
                f"clinical_progression_{infectious_status}",
                STATE[f"subclin_{infectious_status}"],
                STATE[f"clin_{infectious_status}"],
                Param("clinical_progression_rate"),
            )
        )
        model.add_flow(
            TransitionFlow(
                f"clinical_regression_{infectious_status}",
                STATE[f"clin_{infectious_status}"],
                STATE[f"subclin_{infectious_status}"],
                Param("clinical_regression_rate"),
            )
        )
    # Infectiousness onset and loss (the original's flow names carry the typo "infectiousnnes")
    for clinical_status in ["clin", "subclin"]:
        model.add_flow(
            TransitionFlow(
                f"infectiousnnes_gain_{clinical_status}",
                STATE[f"{clinical_status}_lowinf"],
                STATE[f"{clinical_status}_inf"],
                Param("infectiousness_gain_rate"),
            )
        )
        model.add_flow(
            TransitionFlow(
                f"infectiousnnes_loss_{clinical_status}",
                STATE[f"{clinical_status}_inf"],
                STATE[f"{clinical_status}_lowinf"],
                Param("infectiousness_loss_rate"),
            )
        )
    # TB self-recovery
    for infectious_status in ["inf", "lowinf"]:
        model.add_flow(
            TransitionFlow(
                f"self_recovery_{infectious_status}",
                STATE[f"subclin_{infectious_status}"],
                STATE["recovered"],
                Param("self_recovery_rate"),
            )
        )
    return model, infection_flows


def get_force_of_infection(
    contact_rate: Any, infection_pop_scale: Any, age_mixing_matrix: Any = None
) -> ForceOfInfection:
    """Generalised force of infection ``M(t) @ (I_w / N ** infection_pop_scale)``.

    summer2's ``add_infection_generalised_flow`` plus, from ``stratify_model_by_age``,
    ``set_mixing_matrix`` and ``add_infectiousness_adjustments``. Infectiousness is reduced for
    less infectious and subclinical disease and is zero under 15. Without a mixing matrix the
    whole population is one mixing pool (``group_by=None``), as in summer2's unstratified
    infection.
    """
    infectiousness = [
        (STATE.isin(["subclin_lowinf", "clin_lowinf"]), Param("rel_infectiousness_lowinf")),
        (STATE.isin(["subclin_lowinf", "subclin_inf"]), Param("rel_infectiousness_subclin")),
        (CHILDREN, 0.0),
    ]
    by_age: dict[str, Any] = {}
    if age_mixing_matrix is not None:
        by_age = {
            "group_by": AGE,
            "mixing": MixingMatrix(AGE, age_mixing_matrix, normalize="none"),
        }
    return ForceOfInfection(
        "infection",
        infectious=STATE.isin(INFECTIOUS_COMPARTMENTS),
        kind=FOIKind.GENERALISED,
        exponent=infection_pop_scale,
        contact_rate=contact_rate,
        infectiousness=infectiousness,
        **by_age,
    )


def tanh_based_scaleup(
    t: Any,
    shape: Any,
    inflection_time: Any,
    start_asymptote: Any,
    end_asymptote: Any = 1.0,
) -> Any:
    """``(tanh(shape (t - inflection_time)) / 2 + 0.5) (end - start) + start``.

    The original's function, unchanged: it is plain arithmetic, so called on rate expressions
    (``Time()``, ``Param``) it returns a rate expression.
    """
    rng = end_asymptote - start_asymptote
    return (tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote


def add_detection_and_treatment(
    model: FlowModel,
    time_variant_tsr: Any,
    screening_programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...],
) -> tuple[list[str], Any]:
    """Passive detection, screening and treatment outcomes.

    Returns the screening flow names and the passive detection rate (the original registers the
    latter as a computed value here).
    """
    tv_detection_rate = Param("recent_detection_rate") * tanh_based_scaleup(
        Time(),
        Param("passive_detection_shape"),
        Param("passive_detection_inflection"),
        Param("passive_detection_past_frac"),  # past detection rate, relative to current
        1.0,
    )
    for active_comp in ACTIVE_COMPS:
        multiplier = Param("rel_detection_subclin") if active_comp.startswith("subclin_") else 1.0
        model.add_flow(
            TransitionFlow(
                f"tb_detection_{active_comp}",
                STATE[active_comp],
                STATE["treatment"],
                multiplier * tv_detection_rate,
            )
        )

    # Screening programs for TBI and TB
    screening_flows = []
    for scr_prog in screening_programs:
        dest_comp = scr_prog.scr_tool["dest_comp"]
        for source_comp, sensitivity in scr_prog.scr_tool["sensitivities"].items():
            flow_name = f"{scr_prog.name}_{source_comp}"
            model.add_flow(
                TransitionFlow(
                    flow_name,
                    STATE[source_comp],
                    STATE[dest_comp],
                    sensitivity * scr_prog.scr_tool["success_prop"] * scr_prog.raw_screening_func,
                )
            )
            screening_flows.append(flow_name)

    # Treatment outcomes: recovery and relapse here; deaths in add_births_and_deaths.
    model.add_flow(
        TransitionFlow(
            "tx_recovery",
            STATE["treatment"],
            STATE["recovered"],
            time_variant_tsr / Param("tx_duration"),
        )
    )
    # Placeholder rate: the age-specific relapse rate is set in stratify_model_by_age.
    model.add_flow(TransitionFlow("tx_relapse", STATE["treatment"], STATE["subclin_lowinf"], 1.0))
    return screening_flows, tv_detection_rate


def stratify_model_by_age(
    model: FlowModel,
    neg_tx_outcome_funcs: Any,
    screening_programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...],
) -> None:
    """Split by age, add ageing, and set the age-specific adjustments.

    summer2's ``AgeStratification`` also adds ageing between consecutive bands at one over the
    band width; summer4 declares that as one flow with a ``TraitChain`` pairing.
    """
    assert "15" in AGE_GROUPS, "15 must be an age break (BCG effect and mixing matrix)"
    assert "0" in AGE_GROUPS and "5" in AGE_GROUPS, "for the age-specific infection parameters"

    model.stratify(AGE)
    model.add_flow(
        TransitionFlow(
            "ageing", AGE.present(), AGE.present(), 1.0, pairing=TraitChain.from_breakpoints(AGE)
        )
    )

    # Adjust infection progression and containment by age
    for flow_name in ["progression_lowinf", "progression_inf", "containment"]:
        param_stem = "containment" if flow_name == "containment" else "progression"
        adjustments = []
        for age in AGE_GROUPS:
            int_age = int(age)
            latency_age_cat = "age0" if int_age < 5 else "age5" if int_age < 15 else "age15"
            adjustments.append(
                Multiply(Param(f"{param_stem}_rate_{latency_age_cat}"), where=AGE[age])
            )
        model.adjust_flow(flow_name, *adjustments)

    # Children's susceptibility to infection (BCG effect)
    model.adjust_flow(
        "infection_from_mtb_naive", Multiply(Param("rel_sus_children"), where=CHILDREN)
    )

    # Treatment outcomes are age-specific because natural mortality is
    model.adjust_flow("tx_relapse", Multiply(neg_tx_outcome_funcs.relapse))

    # Screening interventions by age
    for scr_prog in screening_programs:
        age_multipliers = scr_prog.strata_coverage_multipliers.get("age", {})
        assert set(age_multipliers).issubset(AGE_GROUPS), "screening ages must be model ages"
        if not age_multipliers:
            continue
        for flow_name in scr_prog.flow_names():
            model.adjust_flow(
                flow_name, *(Multiply(m, where=AGE[age]) for age, m in age_multipliers.items())
            )


def stratify_model_by_reachability(
    model: FlowModel, infection_flows: list[str], screening_flows: list[str]
) -> None:
    """Split into the reachable and the unreachable.

    The unreachable have no access to screening, a higher infection risk and less access to
    passive case finding. The split is constant over time: births are split by
    ``reachable_pop_frac`` and deaths return to the stratum they leave. Mixing between the two
    strata is homogeneous.
    """
    model.stratify(REACH)
    unreachable = REACH["unreachable"]
    for flow in infection_flows:
        model.adjust_flow(flow, Multiply(Param("rel_sus_unreachable"), where=unreachable))
    for flow in [f"tb_detection_{comp}" for comp in ACTIVE_COMPS]:
        model.adjust_flow(flow, Multiply(Param("rel_detection_unreachable"), where=unreachable))
    for flow in screening_flows:
        model.adjust_flow(flow, Multiply(0.0, where=unreachable))


def add_births_and_deaths(
    model: FlowModel, agg_pop_data: pd.Series, bckd_death_funcs: Any, neg_tx_outcome_funcs: Any
) -> tuple[dict[str, tuple[str, Any]], dict[str, tuple[str, Any]]]:
    """Deaths, recycled to (``mtb_naive``, youngest age) in the same reachability; births.

    Returns the original's natural and TB death "flows" as ``{output name: (flow, source)}``:
    the original declares one flow per (cause, compartment, age) and names outputs after them;
    here each cause is one flow, and those names select its sources.
    """
    # All-cause (non-TB) mortality: every compartment except the one deaths go to.
    model.add_flow(TransitionFlow("all_cause_mortality", ~NEWBORN, NEWBORN, bckd_death_funcs))
    nat_death_flows = {}
    for compartment in COMPARTMENTS:
        for i_age, source_age in enumerate(AGE_GROUPS):
            if i_age > 0 or compartment != "mtb_naive":
                name = f"all_cause_mortality_from_{compartment}_age_{source_age}"
                source = STATE[compartment] & AGE[source_age]
                nat_death_flows[name] = ("all_cause_mortality", source)

    # Deaths caused by untreated TB, only from clinical TB
    tb_death_flows = {}
    for infectious_status in ["inf", "lowinf"]:
        flow = f"tb_mortality_{infectious_status}"
        model.add_flow(
            TransitionFlow(
                flow,
                STATE[f"clin_{infectious_status}"],
                NEWBORN,
                Param(f"tb_mortality_rate_{infectious_status}"),
            )
        )
        for source_age in AGE_GROUPS:
            tb_death_flows[f"{flow}_age_{source_age}"] = (flow, AGE[source_age])
    # Deaths on TB treatment
    model.add_flow(
        TransitionFlow("tx_death", STATE["treatment"], NEWBORN, neg_tx_outcome_funcs.death)
    )
    for source_age in AGE_GROUPS:
        tb_death_flows[f"tx_death_age_{source_age}"] = ("tx_death", AGE[source_age])

    # Extra births for population growth: yearly increments (missing years filled linearly),
    # zero growth the year before the first, sigmoidally interpolated.
    full_index = pd.Index(range(agg_pop_data.index.min(), agg_pop_data.index.max() + 1))
    pop_entry = agg_pop_data.reindex(full_index).interpolate().diff().dropna()
    entry_rate = Data(
        [pop_entry.index.min() - 1.0] + pop_entry.index.to_list(), [0.0] + pop_entry.to_list()
    ).interp("sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE)
    frac = Param("reachable_pop_frac")
    for reach, share in (("reachable", frac), ("unreachable", 1.0 - frac)):
        model.add_flow(EntryFlow(f"births_{reach}", NEWBORN & REACH[reach], entry_rate * share))
    return nat_death_flows, tb_death_flows


class NegTxOutcomes(NamedTuple):
    """Per-age negative treatment outcome rates (per year on treatment)."""

    relapse: Any
    death: Any


def get_neg_tx_outcomes(
    bckd_death_rate: Any, time_variant_tsr: Any, tx_duration: Any, pct_neg_tx_death: Any
) -> NegTxOutcomes:
    """The original's ``get_neg_tx_outcome_funcs_age``, for every age at once.

    ``bckd_death_rate`` arrives as summer4's per-age ``GroupedRate``, which supports NumPy
    functions (``np.exp``, ``np.maximum``) and arithmetic but not ``jnp`` functions directly.

    Natural death on treatment is ``1 - exp(-tx_duration * mu)``. Deaths caused by TB on
    treatment take the requested share of unsuccessful outcomes left after natural deaths,
    floored at zero; relapse takes the rest.
    """
    prop_natural_death_while_on_treatment = 1.0 - np.exp(-tx_duration * bckd_death_rate)
    requested_prop_death_on_treatment = (1.0 - time_variant_tsr) * pct_neg_tx_death / 100.0
    prop_death_from_treatment = np.maximum(
        requested_prop_death_on_treatment - prop_natural_death_while_on_treatment, 0.0
    )
    relapse_prop = (
        1.0 - time_variant_tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    )
    return NegTxOutcomes(
        relapse=relapse_prop / tx_duration, death=prop_death_from_treatment / tx_duration
    )


def get_neg_tx_outcome_funcs(bckd_death_funcs: Any, time_variant_tsr: Any) -> NegTxOutcomes:
    """Relapse and on-treatment death rates as two rate expressions computed together.

    summer2 builds one ``Function`` per age returning a dict. Here one ``defer`` of
    :func:`get_neg_tx_outcomes` takes the per-age death-rate table and returns both rates for
    every age; ``field_refs`` names its two fields. The function runs once per vector-field call
    however many flows read it.
    """
    computed = defer(get_neg_tx_outcomes)(
        bckd_death_funcs, time_variant_tsr, Param("tx_duration"), Param("pct_neg_tx_death")
    )
    return field_refs(computed, NegTxOutcomes)
