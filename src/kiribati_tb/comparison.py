"""Parity against improved: what changes, by how much, and which is closer to the truth.

Used by ``scripts/compare_improved.py`` and ``docs/08-parity-and-improved.md``.

Two separate questions:

1. **The solver.** :func:`solver_comparison` runs the same model with several solver settings
   and measures each against a tight reference solve (Dormand–Prince at ``1e-12`` with the
   yearly mixing jumps given to the controller).
2. **What a flow output means.** A yearly flow output should stand for the number of people
   who made the transition during the year. :func:`annual_flow_comparison` computes that number
   accurately (the integral of the instantaneous rate saved every 0.02 years) and sets the
   parity value (summer2's midpoint: the average of the rate at this year's and last year's
   save times) and the improved values beside it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import diffrax
import numpy as np
import pandas as pd

from summer4 import Compartments, OutputSet, SavePlan
from summer4.solvers import Diffrax

from kiribati_tb.model import TBModel, get_tb_model
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors
from kiribati_tb.variants import END_TIME, IMPROVED, MIXING_JUMPS, PARITY, START_TIME, Variant

YEARS = np.arange(START_TIME, END_TIME + 1.0)
FINE_STEP = 0.02

SOLVERS: dict[str, dict[str, Any]] = {
    "parity: Dopri5, rtol=atol=1e-10": PARITY.run_kwargs(),
    "summer2 default: Dopri5, rtol=atol=1.4e-4": {
        "solver": "dopri5",
        "rtol": 1.4e-4,
        "atol": 1.4e-4,
    },
    "summer4 default: Tsit5, rtol=1e-6, atol=1e-9": {"max_steps": 16384},
    "improved: Tsit5, rtol=1e-6, atol=1e-9, jump_ts": IMPROVED.run_kwargs(),
}
REFERENCE_SOLVER: dict[str, Any] = {
    "solver": Diffrax(
        diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12, jump_ts=MIXING_JUMPS),
    ),
    "max_steps": 200_000,
}


def build(
    variant: Variant, scenario: str | None = None, heterogeneous_mixing: bool = True
) -> TBModel:
    """The model for one variant and (optionally) one screening scenario."""
    from kiribati_tb.scenarios import SCENARIOS_BY_ID

    _, _, tv_params = get_parameters_and_priors()
    config = DEFAULT_MODEL_CONFIG | {"heterogeneous_mixing": heterogeneous_mixing}
    programs = SCENARIOS_BY_ID[scenario].scr_prgs if scenario else []
    return get_tb_model(config, tv_params, programs, variant)


def _solve_compartments(
    tb: TBModel, params: Mapping[str, Any], run_kwargs: Mapping[str, Any]
) -> tuple[np.ndarray, int, int]:
    outputs = OutputSet()
    outputs["y"] = Compartments()
    plan = outputs.plan(SavePlan(ts=YEARS))

    def solve() -> Any:
        return tb.model.run(params, t0=START_TIME, t1=END_TIME, dt=1.0, save=plan, **run_kwargs)

    result = solve()
    values = np.asarray(result["y"].values.data)
    info = result.solver
    steps = int(info.num_steps) if info is not None and info.num_steps is not None else -1
    rejected = (
        int(info.num_rejected_steps)
        if info is not None and info.num_rejected_steps is not None
        else -1
    )
    return values, steps, rejected


def solver_comparison(tb: TBModel, params: Mapping[str, Any]) -> pd.DataFrame:
    """Each solver's error against the tight reference, and its step counts.

    The error is the relative difference over compartments of at least one person and over the
    years 1850–2035. Steps (accepted plus rejected) are the machine-independent cost: each is a
    fixed number of vector-field evaluations (6 for Dopri5 and Tsit5 with FSAL).
    """
    reference, _, _ = _solve_compartments(tb, params, REFERENCE_SOLVER)
    mask = np.abs(reference) >= 1.0
    rows = []
    for label, kwargs in SOLVERS.items():
        values, steps, rejected = _solve_compartments(tb, params, kwargs)
        rel = np.abs(values - reference)[mask] / np.abs(reference)[mask]
        rows.append(
            {
                "solver": label,
                "steps": steps,
                "max_rel_error": float(rel.max()),
                "median_rel_error": float(np.median(rel)),
                "rejected": rejected,
            }
        )
    return pd.DataFrame(rows).set_index("solver")


ANNUAL_FLOWS: tuple[str, ...] = ("tb_incidence", "notifications", "tb_mortality", "screening")


def annual_flow_comparison(
    parity_tb: TBModel,
    improved_tb: TBModel,
    params: Mapping[str, Any],
    names: Sequence[str] = ANNUAL_FLOWS,
    years: Sequence[float] = tuple(np.arange(2020.0, 2036.0)),
) -> pd.DataFrame:
    """Yearly flow outputs three ways, against the accurate number of people per year.

    Columns, for each output and year ``t``:

    - ``accurate``: people making the transition during ``(t - 1, t]``, the integral of the
      instantaneous rate saved every ``FINE_STEP`` years and at the screening ramp times
      (trapezoid), solved with the improved solver.
    - ``parity``: the ``PARITY`` output (summer2 midpoint), solved at the goldens' tolerance.
    - ``improved_rate``: the ``IMPROVED`` output at ``t``: the instantaneous rate, people/year.
    - ``improved_interval``: the integral over ``(t - 1, t]`` from the improved variant's own
      grid (0.1 years plus the ramp times), the yearly differences of its ``cumulative_integral``.
    """
    from kiribati_tb.runner_tools import run_model

    parity = run_model(parity_tb, params, PARITY, names=list(names))
    improved = run_model(improved_tb, params, IMPROVED, names=list(names))

    fine_outputs = improved_tb.outputs.subset(list(names))
    fine = IMPROVED.with_save_step(FINE_STEP)
    fine_frame = run_model(improved_tb, params, fine, outputs=fine_outputs)

    def per_year(frame: pd.DataFrame, name: str) -> pd.Series:
        t = frame.index.to_numpy()
        f = frame[name].to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
        at = np.interp(np.asarray(years), t, cumulative)
        before = np.interp(np.asarray(years) - 1.0, t, cumulative)
        return pd.Series(at - before, index=pd.Index(years, name="year"))

    pieces = []
    for name in names:
        frame = pd.DataFrame(
            {
                "accurate": per_year(fine_frame, name),
                "parity": parity[name].reindex(years).to_numpy(),
                "improved_rate": improved[name].reindex(years, method="nearest").to_numpy(),
                "improved_interval": per_year(improved, name).to_numpy(),
            },
            index=pd.Index(years, name="year"),
        )
        frame.insert(0, "output", name)
        pieces.append(frame)
    out = pd.concat(pieces).reset_index().set_index(["output", "year"])
    scale = out["accurate"].abs().clip(lower=1e-9)
    for column in ("parity", "improved_rate", "improved_interval"):
        out[f"{column}_err_%"] = 100.0 * (out[column] - out["accurate"]) / scale
    return out


def cumulative_comparison(
    parity_tb: TBModel, improved_tb: TBModel, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Cumulative incidence and mortality from 2026, both variants, at whole years."""
    names = ["cum_tb_incidence", "cum_tb_mortality"]
    from kiribati_tb.runner_tools import run_model

    parity = run_model(parity_tb, params, PARITY, names=names)
    improved = run_model(improved_tb, params, IMPROVED, names=names)
    years = np.arange(2026.0, 2036.0)
    rows = {}
    for name in names:
        rows[f"{name} (parity: cumsum of midpoints)"] = parity[name].reindex(years).to_numpy()
        rows[f"{name} (improved: cumulative_integral)"] = (
            improved[name].reindex(years, method="nearest").to_numpy()
        )
    return pd.DataFrame(rows, index=pd.Index(years, name="year"))
