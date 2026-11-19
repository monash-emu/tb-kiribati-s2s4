"""The same Kiribati TB model, written the summer4 way.

:mod:`kiribati_tb.model` follows ``tbh/model.py`` function for function, so a summer2 user can
read the two side by side. This module builds the identical model (same compartments, rates and
trajectories; it passes the same golden tests) in compact summer4 idiom:

- **The map first.** ``STATE × AGE × REACH`` is declared up front, and each flow carries its
  own adjustments, instead of stratifying later and adjusting flows by name.
- **One flow over a selector** wherever the original has one flow per compartment with the same
  rate, or rates that differ by a per-compartment factor: infection (4 flows → 1), detection
  (4 → 1), self-recovery (2 → 1), TB mortality (2 → 1), each screening program (one flow per
  detected state → 1), births (2 → 1, with a parameter ``split``).
- **``TraitChain`` within the state property** for moves between paired states: clinical
  progression and regression, infectiousness gain and loss (2 → 1 each), and progression from
  ``incipient`` to the two subclinical states (2 → 1, the infectious share as the per-pair ``rates``).

Outputs keep the original names: :func:`flow_sources` maps every original flow name that no
longer exists to ``(flow, source, dest)``, and :func:`~kiribati_tb.outputs.request_model_outputs`
selects it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from summer4 import (
    EntryFlow,
    FlowModel,
    InitialPopulation,
    Multiply,
    Param,
    PropertyMap,
    REMAINDER,
    Split,
    Time,
    TraitChain,
    TransitionFlow,
    defer,
)
from summer4.data import Data

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
    NEWBORN,
    REACH,
    STATE,
    SUMMER2_SIGMOID_CURVATURE,
    TBModel,
    get_force_of_infection,
    get_neg_tx_outcome_funcs,
    tanh_based_scaleup,
)
from kiribati_tb.outputs import request_model_outputs
from kiribati_tb.variants import PARITY, Variant

SUBCLINICAL = STATE.isin(["subclin_lowinf", "subclin_inf"])
UNREACHABLE = REACH["unreachable"]
LATENCY_BANDS = {"age0": ("0", "3"), "age5": ("5", "10"), "age15": ("15", "18", "40", "65")}

# Moves between paired states: one TraitChain over STATE each.
CLINICAL = (("subclin_inf", "clin_inf"), ("subclin_lowinf", "clin_lowinf"))
INFECTIOUSNESS = (("subclin_lowinf", "subclin_inf"), ("clin_lowinf", "clin_inf"))
PROGRESSION = (("incipient", "subclin_lowinf"), ("incipient", "subclin_inf"))


def reverse(pairs: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """The same pairs, destination to source."""
    return tuple((dest, src) for src, dest in pairs)


def chain(
    name: str,
    pairs: tuple[tuple[str, str], ...],
    rate: Any,
    *,
    rates: tuple[Any, ...] | None = None,
    **kw: Any,
) -> TransitionFlow:
    """One flow over explicit state pairs, within every age and reachability stratum.

    ``rates`` are per-pair multipliers: numbers or rate expressions.
    """
    pairing = TraitChain(STATE, pairs, rates=rates)
    return TransitionFlow(name, STATE.present(), STATE.present(), rate, pairing=pairing, **kw)


def by_latency_band(stem: str) -> list[Multiply]:
    """``Param(f"{stem}_rate_{band}")`` for under 5, 5–14 and 15 and over."""
    return [
        Multiply(Param(f"{stem}_rate_{band}"), where=AGE.isin(ages))
        for band, ages in LATENCY_BANDS.items()
    ]


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

    model = FlowModel(PropertyMap.from_property(STATE).stratify(AGE).stratify(REACH))
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
        # Early infection
        TransitionFlow(
            "containment",
            s["incipient"],
            s["contained"],
            1.0,
            adjust=by_latency_band("containment"),
        ),
        TransitionFlow("clearance", s["contained"], s["cleared"], Param("clearance_rate")),
        TransitionFlow("breakdown", s["contained"], s["incipient"], Param("breakdown_rate")),
        # One source, two destinations: the infectious share as per-pair rates
        chain(
            "progression",
            PROGRESSION,
            1.0,
            rates=(1.0 - p_inf, p_inf),
            adjust=by_latency_band("progression"),
        ),
        # Active TB: paired moves within the state property
        chain("clinical_progression", CLINICAL, Param("clinical_progression_rate")),
        chain("clinical_regression", reverse(CLINICAL), Param("clinical_regression_rate")),
        chain("infectiousness_gain", INFECTIOUSNESS, Param("infectiousness_gain_rate")),
        chain("infectiousness_loss", reverse(INFECTIOUSNESS), Param("infectiousness_loss_rate")),
        TransitionFlow("self_recovery", SUBCLINICAL, s["recovered"], Param("self_recovery_rate")),
        # Passive detection and treatment
        TransitionFlow(
            "tb_detection",
            s.isin(ACTIVE_COMPS),
            s["treatment"],
            detection,
            adjust=[
                Multiply(Param("rel_detection_subclin"), where=SUBCLINICAL),
                Multiply(Param("rel_detection_unreachable"), where=UNREACHABLE),
            ],
        ),
        TransitionFlow(
            "tx_recovery", s["treatment"], s["recovered"], success / Param("tx_duration")
        ),
        TransitionFlow("tx_relapse", s["treatment"], s["subclin_lowinf"], outcomes.relapse),
        # Ageing, deaths (recycled to the youngest age in the same reachability), births
        TransitionFlow(
            "ageing", AGE.present(), AGE.present(), 1.0, pairing=TraitChain.from_breakpoints(AGE)
        ),
        TransitionFlow("all_cause_mortality", ~NEWBORN, NEWBORN, death),
        TransitionFlow(
            "tb_mortality",
            s.isin(["clin_inf", "clin_lowinf"]),
            NEWBORN,
            1.0,
            adjust=[
                Multiply(Param(f"tb_mortality_rate_{inf}"), where=s[f"clin_{inf}"])
                for inf in ("inf", "lowinf")
            ],
        ),
        TransitionFlow("tx_death", s["treatment"], NEWBORN, outcomes.death),
        EntryFlow(
            "births", NEWBORN, entry, split={REACH: {"reachable": frac, "unreachable": 1.0 - frac}}
        ),
        *screening_flows(screening_programs),
    ]:
        model.add_flow(flow)

    model.set_initial_population(
        InitialPopulation(
            {
                s["mtb_naive"]: pop.iloc[0] - model_config["seed"],
                s["clin_inf"]: model_config["seed"],
            },
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
    )
    return TBModel(model.compile(), outputs, screening_breakpoints(screening_programs))


def screening_breakpoints(
    programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...],
) -> tuple[float, ...]:
    """Times where a screening rate has a kink: each campaign's 0.01-year ramps."""
    return tuple(
        sorted(
            {
                float(t)
                for p in programs
                for t in (p.start_time - 0.01, p.start_time, p.end_time - 0.01, p.end_time)
            }
        )
    )


def screening_flows(
    programs: list[ScreeningProgram] | tuple[ScreeningProgram, ...],
) -> list[TransitionFlow]:
    """One flow per program from every state its tool detects; sensitivity per state."""
    flows = []
    for prog in programs:
        tool = prog.scr_tool
        ages = prog.strata_coverage_multipliers.get("age", {})
        flows.append(
            TransitionFlow(
                prog.name,
                STATE.isin(list(tool["sensitivities"])),
                STATE[tool["dest_comp"]],
                tool["success_prop"] * prog.raw_screening_func,
                adjust=[
                    *(Multiply(se, where=STATE[c]) for c, se in tool["sensitivities"].items()),
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
    for inf in ("inf", "lowinf"):
        sources[f"progression_{inf}"] = ("progression", None, STATE[f"subclin_{inf}"])
    for comp in ACTIVE_COMPS:
        sources[f"tb_detection_{comp}"] = ("tb_detection", STATE[comp], None)
    for prog in programs:
        for comp in prog.scr_tool["sensitivities"]:
            sources[f"{prog.name}_{comp}"] = (prog.name, STATE[comp], None)
    return sources


def death_outputs() -> tuple[dict[str, tuple[str, Any]], dict[str, tuple[str, Any]]]:
    """The original's per-(compartment, age) death output names, as flow sources."""
    natural = {
        f"all_cause_mortality_from_{comp}_age_{age}": (
            "all_cause_mortality",
            STATE[comp] & AGE[age],
        )
        for comp in COMPARTMENTS
        for i, age in enumerate(AGE_GROUPS)
        if i > 0 or comp != "mtb_naive"
    }
    tb: dict[str, tuple[str, Any]] = {
        f"tb_mortality_{inf}_age_{age}": ("tb_mortality", STATE[f"clin_{inf}"] & AGE[age])
        for inf in ("inf", "lowinf")
        for age in AGE_GROUPS
    }
    tb |= {f"tx_death_age_{age}": ("tx_death", AGE[age]) for age in AGE_GROUPS}
    return natural, tb
