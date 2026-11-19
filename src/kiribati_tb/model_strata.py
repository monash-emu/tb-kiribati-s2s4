"""The same Kiribati TB model with TB disease as a state stratified by two properties.

The original has four disease states, ``subclin_lowinf``, ``clin_lowinf``, ``subclin_inf`` and
``clin_inf``: two axes, clinical status and infectiousness, spelled into state names. Here
there is one disease state, ``tb``, stratified (only it) by

- ``CLIN`` = (``subclin``, ``clin``), and
- ``INF`` = (``lowinf``, ``inf``).

The compartments are the same 160 (``state=tb, clin=subclin, inf=lowinf`` is the original's
``subclin_lowinf``), and so are the equations. What changes is that each axis is its own
property:

- clinical progression is ``CLIN["subclin"] → CLIN["clin"]`` and infectiousness gain is
  ``INF["lowinf"] → INF["inf"]``; the other axis, age and reachability are carried along;
- factors name the axis they depend on (``where=CLIN["subclin"]``, ``where=INF["lowinf"]``);
- progression from ``incipient`` lands in ``CLIN["subclin"]`` with the infectious share as a
  parameter ``split`` over ``INF``, the destination-only property.

Every flow into ``tb`` names its ``CLIN × INF`` cell (progression, relapse, the seed). Without
that, summer4 would share people equally among the four disease compartments.

The output names stay the original's: :func:`compartment` maps each original state name to its
selector, and :func:`flow_sources` each original flow name to the flow that now carries it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from summer4 import (
    REMAINDER,
    EntryFlow,
    FlowModel,
    InitialPopulation,
    Multiply,
    Param,
    Property,
    PropertyMap,
    Selector,
    Split,
    Time,
    TraitChain,
    TransitionFlow,
    defer,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix

from kiribati_tb.age_mixing import canberra_distance, get_model_ready_age_mixing_matrix
from kiribati_tb.age_mixing import read_conmat_matrix
from kiribati_tb.demographic_tools import AGE_GROUPS, load_demography
from kiribati_tb.interventions import ScreeningProgram
from kiribati_tb.model import (
    ACTIVE_COMPS,
    AGE,
    CHILDREN,
    COMPARTMENTS,
    LATENT_COMPS,
    REACH,
    SUMMER2_SIGMOID_CURVATURE,
    TBModel,
    get_neg_tx_outcome_funcs,
    tanh_based_scaleup,
)
from kiribati_tb.model_summer4 import UNREACHABLE, by_latency_band, screening_breakpoints
from kiribati_tb.outputs import request_model_outputs
from kiribati_tb.variants import PARITY, Variant

STATE = Property(
    "state", ("mtb_naive", "incipient", "contained", "cleared", "tb", "treatment", "recovered")
)
CLIN = Property("clin", ("subclin", "clin"))
INF = Property("inf", ("lowinf", "inf"))
TB = STATE["tb"]
NEWBORN = STATE["mtb_naive"] & AGE[AGE_GROUPS[0]]


def compartment(name: str) -> Selector:
    """The selector for one of the original's ten state names."""
    if name in ACTIVE_COMPS:
        clin, inf = name.split("_")
        return TB & CLIN[clin] & INF[inf]
    return STATE[name]


def get_force_of_infection(contact_rate: Any, exponent: Any, mixing: Any = None) -> Any:
    """The generalised force of infection; infectiousness by axis, zero under 15."""
    by_age: dict[str, Any] = {}
    if mixing is not None:
        by_age = {"group_by": AGE, "mixing": MixingMatrix(AGE, mixing, normalize="none")}
    return ForceOfInfection(
        "infection",
        infectious=TB,
        kind=FOIKind.GENERALISED,
        exponent=exponent,
        contact_rate=contact_rate,
        infectiousness=[
            (INF["lowinf"], Param("rel_infectiousness_lowinf")),
            (CLIN["subclin"], Param("rel_infectiousness_subclin")),
            (CHILDREN, 0.0),
        ],
        **by_age,
    )


def get_tb_model(
    model_config: dict,
    tv_params: dict[str, pd.Series],
    screening_programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...] = (),
    variant: Variant = PARITY,
) -> TBModel:
    """Build the model and its outputs; the same call and result as ``model.get_tb_model``."""
    if tuple(model_config["age_groups"]) != AGE_GROUPS:
        raise ValueError(f"This port is built for age groups {AGE_GROUPS}")
    demog = load_demography(model_config["iso3"], AGE_GROUPS, model_config["pop_scaling"])
    pop = demog.total
    death = demog.death_rate_table(AGE).interp("sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE)
    tsr = tv_params["tx_success_pct"]
    success = Data(tsr.index.to_numpy(float), tsr.to_numpy(float) / 100.0).interp("linear")
    outcomes = get_neg_tx_outcome_funcs(death, success)
    entry = Data(demog.entry_times, demog.entry_values).interp(
        "sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE
    )
    hetero = model_config.get("heterogeneous_mixing", True)
    mixing = get_model_ready_age_mixing_matrix() if hetero else None
    exponent = Param("infection_pop_scale")
    foi = get_force_of_infection(
        Param("raw_transmission_rate") * pop.loc[2020] ** exponent, exponent, mixing
    )
    detection = Param("recent_detection_rate") * tanh_based_scaleup(
        Time(),
        Param("passive_detection_shape"),
        Param("passive_detection_inflection"),
        Param("passive_detection_past_frac"),
    )
    p_inf, frac = Param("progression_prop_infectious"), Param("reachable_pop_frac")
    s = STATE

    pmap = (
        PropertyMap.from_property(STATE)
        .stratify(CLIN, where=TB)
        .stratify(INF, where=TB)
        .stratify(AGE)
        .stratify(REACH)
    )
    model = FlowModel(pmap)
    for flow in [
        # Infection and reinfection, with each source state's relative susceptibility
        TransitionFlow(
            "infection",
            s.isin(["mtb_naive", "contained", "cleared", "recovered"]),
            s["incipient"],
            foi,
            adjust=[
                Multiply(Param("rel_sus_children"), where=s["mtb_naive"] & CHILDREN),
                Multiply(Param("rel_sus_contained"), where=s["contained"]),
                Multiply(Param("rel_sus_cleared"), where=s.isin(["cleared", "recovered"])),
                Multiply(Param("rel_sus_unreachable"), where=UNREACHABLE),
            ],
        ),
        # Early infection; progression lands in subclinical disease, split by infectiousness
        TransitionFlow(
            "containment",
            s["incipient"],
            s["contained"],
            1.0,
            adjust=by_latency_band("containment"),
        ),
        TransitionFlow("clearance", s["contained"], s["cleared"], Param("clearance_rate")),
        TransitionFlow("breakdown", s["contained"], s["incipient"], Param("breakdown_rate")),
        TransitionFlow(
            "progression",
            s["incipient"],
            TB & CLIN["subclin"],
            1.0,
            split={INF: {"lowinf": 1.0 - p_inf, "inf": p_inf}},
            adjust=by_latency_band("progression"),
        ),
        # Active TB: each axis moves on its own
        TransitionFlow(
            "clinical_progression",
            CLIN["subclin"],
            CLIN["clin"],
            Param("clinical_progression_rate"),
        ),
        TransitionFlow(
            "clinical_regression", CLIN["clin"], CLIN["subclin"], Param("clinical_regression_rate")
        ),
        TransitionFlow(
            "infectiousness_gain", INF["lowinf"], INF["inf"], Param("infectiousness_gain_rate")
        ),
        TransitionFlow(
            "infectiousness_loss", INF["inf"], INF["lowinf"], Param("infectiousness_loss_rate")
        ),
        TransitionFlow(
            "self_recovery", CLIN["subclin"], s["recovered"], Param("self_recovery_rate")
        ),
        # Passive detection and treatment
        TransitionFlow(
            "tb_detection",
            TB,
            s["treatment"],
            detection,
            adjust=[
                Multiply(Param("rel_detection_subclin"), where=CLIN["subclin"]),
                Multiply(Param("rel_detection_unreachable"), where=UNREACHABLE),
            ],
        ),
        TransitionFlow(
            "tx_recovery", s["treatment"], s["recovered"], success / Param("tx_duration")
        ),
        TransitionFlow(
            "tx_relapse", s["treatment"], compartment("subclin_lowinf"), outcomes.relapse
        ),
        # Ageing, deaths (recycled to the youngest age in the same reachability), births
        TransitionFlow(
            "ageing", AGE.present(), AGE.present(), 1.0, pairing=TraitChain.from_breakpoints(AGE)
        ),
        TransitionFlow("all_cause_mortality", ~NEWBORN, NEWBORN, death),
        TransitionFlow(
            "tb_mortality",
            CLIN["clin"],
            NEWBORN,
            1.0,
            adjust=[
                Multiply(Param(f"tb_mortality_rate_{inf}"), where=INF[inf]) for inf in INF.traits
            ],
        ),
        TransitionFlow("tx_death", s["treatment"], NEWBORN, outcomes.death),
        EntryFlow(
            "births", NEWBORN, entry, split={REACH: {"reachable": frac, "unreachable": 1.0 - frac}}
        ),
        *screening_flows(screening_programs),
    ]:
        model.add_flow(flow)

    seed = model_config["seed"]
    model.set_initial_population(
        InitialPopulation(
            {s["mtb_naive"]: pop.iloc[0] - seed, compartment("clin_inf"): seed},
            splits=(Split(REACH, {"reachable": frac, "unreachable": REMAINDER}),),
        )
    )

    computed_values = {
        "passive_detection_rate_clin": detection,
        "passive_detection_rate_subclin": Param("rel_detection_subclin") * detection,
    }
    if mixing is not None:
        conmat = read_conmat_matrix(model_config["iso3"], AGE_GROUPS)
        computed_values["mixing_matrix_distance"] = defer(canberra_distance)(mixing, conmat)
    nat_deaths, tb_deaths = death_outputs()
    outputs = request_model_outputs(
        COMPARTMENTS,
        ACTIVE_COMPS,
        LATENT_COMPS,
        nat_deaths,
        tb_deaths,
        [name for prog in screening_programs for name in prog.flow_names()],
        computed_values,
        variant,
        flow_sources=flow_sources(screening_programs),
        compartment=compartment,
    )
    return TBModel(model.compile(), outputs, screening_breakpoints(screening_programs))


def screening_flows(
    programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...],
) -> list[TransitionFlow]:
    """One flow per program from every compartment its tool detects; sensitivity per state."""
    flows = []
    for prog in programs:
        tool = prog.scr_tool
        sources = list(tool["sensitivities"])
        source = compartment(sources[0])
        for name in sources[1:]:
            source = source | compartment(name)
        ages = prog.strata_coverage_multipliers.get("age", {})
        flows.append(
            TransitionFlow(
                prog.name,
                source,
                compartment(tool["dest_comp"]),
                tool["success_prop"] * prog.raw_screening_func,
                adjust=[
                    *(
                        Multiply(se, where=compartment(c))
                        for c, se in tool["sensitivities"].items()
                    ),
                    *(Multiply(m, where=AGE[age]) for age, m in ages.items()),
                    Multiply(0.0, where=UNREACHABLE),
                ],
            )
        )
    return flows


def flow_sources(
    programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...] = (),
) -> dict[str, tuple[str, Any, Any]]:
    """Each original flow name that is now part of a larger flow: ``(flow, source, dest)``."""
    sources: dict[str, tuple[str, Any, Any]] = {
        f"births_{reach}": ("births", None, REACH[reach]) for reach in REACH.traits
    }
    for inf in INF.traits:
        sources[f"progression_{inf}"] = ("progression", None, INF[inf])
    for comp in ACTIVE_COMPS:
        sources[f"tb_detection_{comp}"] = ("tb_detection", compartment(comp), None)
    for prog in programs:
        for comp in prog.scr_tool["sensitivities"]:
            sources[f"{prog.name}_{comp}"] = (prog.name, compartment(comp), None)
    return sources


def death_outputs() -> tuple[dict[str, tuple[str, Any]], dict[str, tuple[str, Any]]]:
    """The original's per-(compartment, age) death output names, as flow sources."""
    natural = {
        f"all_cause_mortality_from_{comp}_age_{age}": (
            "all_cause_mortality",
            compartment(comp) & AGE[age],
        )
        for comp in COMPARTMENTS
        for i, age in enumerate(AGE_GROUPS)
        if i > 0 or comp != "mtb_naive"
    }
    tb: dict[str, tuple[str, Any]] = {
        f"tb_mortality_{inf}_age_{age}": ("tb_mortality", compartment(f"clin_{inf}") & AGE[age])
        for inf in INF.traits
        for age in AGE_GROUPS
    }
    tb |= {f"tx_death_age_{age}": ("tx_death", AGE[age]) for age in AGE_GROUPS}
    return natural, tb
