"""Screening tools, programs and scenarios (``tbh/interventions.py``).

The classes keep the original's names, constructor arguments and attributes, so
``data/scenarios.py`` runs unchanged against them (see :mod:`kiribati_tb.scenarios`). The only
difference is the vocabulary of rates: summer2's ``Parameter``, ``Function`` and
``get_linear_interpolation_function`` become summer4's ``Param``, plain arithmetic on rate
expressions, and ``linear(Time(), ...)``.

A screening program screens the reachable population at a constant rate over its campaign,
chosen so that the requested share of the whole population is covered, switched on and off by a
0.01-year linear ramp at each end. Each state a tool detects gets one flow, at the program rate
times the tool's sensitivity for that state times the share successfully started on treatment
(or TPT). ``kiribati_tb.model`` adds those flows and their age and reachability adjustments.
"""

from __future__ import annotations

from typing import Any

from summer4 import Param, Time, log
from summer4.timevarying import linear

TB_STATES = ["subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"]


class ScreeningTools:
    """What each screening tool detects, where the detected go, and the success share."""

    se_ssx = {s: Param(f"prev_se_{s}_ssx") for s in TB_STATES}
    se_cxr = {s: Param(f"prev_se_{s}_cxr") for s in TB_STATES}
    se_plts = {s: Param(f"prev_se_{s}_plts") for s in TB_STATES}
    se_pearl = {s: Param(f"prev_se_{s}_pearl") for s in TB_STATES}

    # TB infection screening and TPT
    TST = {
        "sensitivities": {
            "incipient": Param("prev_se_incipient_tst"),
            "contained": Param("prev_se_contained_tst"),
        },
        "dest_comp": "cleared",
        "success_prop": Param("tpt_completion_perc") / 100.0,  # TPT completion if positive
    }
    # TB disease screening
    SSX = {"sensitivities": se_ssx, "dest_comp": "treatment", "success_prop": 1.0}  # symptoms
    CXR = {"sensitivities": se_cxr, "dest_comp": "treatment", "success_prop": 1.0}  # chest X-ray
    PLTS = {"sensitivities": se_plts, "dest_comp": "treatment", "success_prop": 1.0}  # tongue swab
    PEARL: dict[str, Any]


# PEARL: CXR-Xpert for 35% of those screened, CXR alone for the other 65%.
ScreeningTools.PEARL = {
    "sensitivities": {
        s: 0.35 * ScreeningTools.se_pearl[s] + 0.65 * ScreeningTools.se_cxr[s] for s in TB_STATES
    },
    "dest_comp": "treatment",
    "success_prop": 1.0,
}


def scr_rate_func(total_coverage_perc_among_reachable: Any, duration: float) -> Any:
    """Constant rate that screens the requested share of the reachable over ``duration``."""
    return -log(1.0 - total_coverage_perc_among_reachable / 100.0) / duration


class ScreeningProgram:
    """One screening campaign (the original's constructor and attributes).

    Attributes:
        name: Program name; its flows are named ``{name}_{state}``.
        start_time, end_time: Campaign start and end (years).
        total_coverage_perc_among_reachable: Coverage among the reachable, a rate expression.
        strata_coverage_multipliers: ``{"age": {age: multiplier}}``; unlisted ages 1.
        scr_tool: One of the :class:`ScreeningTools` dicts.
        raw_screening_func: The per-person screening rate as a function of time.
    """

    def __init__(
        self,
        name: str,
        start_time: float,
        end_time: float,
        total_coverage_perc: float,
        strata_coverage_multipliers: dict[str, dict[str, float]],
        scr_tool: dict[str, Any],
    ) -> None:
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.total_coverage_perc_among_reachable = total_coverage_perc / Param("reachable_pop_frac")
        self.strata_coverage_multipliers = strata_coverage_multipliers
        self.scr_tool = scr_tool
        self.get_raw_screening_func()

    def get_raw_screening_func(self) -> None:
        """Set ``raw_screening_func``: the campaign rate, ramped on and off over 0.01 years."""
        duration = self.end_time - self.start_time
        assert duration > 0, "End time must be after Start time"
        scr_rate = scr_rate_func(self.total_coverage_perc_among_reachable, duration)
        s, e = float(self.start_time), float(self.end_time)
        self.raw_screening_func = linear(
            Time(), (s - 0.01, s, e - 0.01, e), (0.0, scr_rate, scr_rate, 0.0)
        )

    def flow_names(self) -> list[str]:
        """The program's flow names, one per detected state."""
        return [f"{self.name}_{comp}" for comp in self.scr_tool["sensitivities"]]


class Scenario:
    """A named set of screening programs, with optional parameter overrides."""

    def __init__(
        self,
        sc_id: str,
        sc_name: str,
        scr_prgs: list[ScreeningProgram],
        sc_name_2: str | None = None,
        desc: str = "",
        params_ow: dict[str, float] | None = None,
    ) -> None:
        self.sc_id = sc_id
        self.sc_name = sc_name
        self.sc_name_2 = sc_name_2 if sc_name_2 is not None else sc_name
        self.scr_prgs = scr_prgs
        self.params_ow = params_ow or {}
        self.description = desc
