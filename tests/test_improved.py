"""The ``IMPROVED`` variant: same model, a cheaper accurate solve, rate-based flow outputs."""

from __future__ import annotations

import numpy as np

from kiribati_tb.comparison import build
from kiribati_tb.runner_tools import run_compartments, run_model
from kiribati_tb.variants import IMPROVED, PARITY
from parity import read_golden, read_params


def test_improved_solver_tracks_the_goldens() -> None:
    """Tsit5 at rtol 1e-6 with the yearly jumps stays within 1e-4 of the summer2 goldens."""
    got = run_compartments(build(IMPROVED), read_params("hetero_baseline"), IMPROVED)
    golden = read_golden("hetero_baseline", "compartments")
    yearly = got.loc[golden.index, golden.columns]
    big = golden.abs() >= 1.0
    rel = ((yearly - golden).abs() / golden.abs())[big]
    assert float(np.nanmax(rel.to_numpy())) < 1e-4


def test_cumulative_outputs_start_at_zero_and_integrate_the_rate() -> None:
    params = read_params("hetero_baseline")
    names = ["tb_incidence", "cum_tb_incidence"]
    improved = run_model(build(IMPROVED), params, IMPROVED, names=names)
    parity = run_model(build(PARITY), params, PARITY, names=names)
    # summer2's cumulative from 2026 already counts the 2025-26 interval; the integral is 0.
    assert improved.loc[2026.0, "cum_tb_incidence"] == 0.0
    np.testing.assert_allclose(
        parity.loc[2026.0, "cum_tb_incidence"], parity.loc[2026.0, "tb_incidence"], rtol=1e-12
    )
    window = improved.loc[2026.0:2027.0, "tb_incidence"]
    trapezoid = np.trapezoid(window.to_numpy(), window.index.to_numpy())
    np.testing.assert_allclose(improved.loc[2027.0, "cum_tb_incidence"], trapezoid, rtol=1e-12)


def test_improved_saves_on_the_screening_ramps() -> None:
    tb = build(IMPROVED, "scenario_3")
    assert tb.breakpoints == (2025.99, 2026.0, 2026.99, 2027.0)
    assert set(tb.breakpoints) <= set(np.round(IMPROVED.save_times(tb.breakpoints), 9))
