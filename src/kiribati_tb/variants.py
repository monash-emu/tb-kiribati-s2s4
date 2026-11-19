"""The two ways to run the model: ``PARITY`` (matches summer2) and ``IMPROVED``.

The model structure is the same in both. They differ only in how the ODE is solved and how flow
outputs are reported:

================================  ===================================  ==================================
Choice                            ``PARITY``                           ``IMPROVED``
================================  ===================================  ==================================
Flow output values                ``"midpoint"`` (summer2's average    ``"rate"``: the instantaneous rate
                                  of a flow's rate at this save time   at the save time, people per year
                                  and the one before)
Cumulative outputs                ``cumsum`` of those amounts, from    ``cumulative_integral`` of the
                                  2026 (counts the 2025–26 interval)   rate from 2026 (0 in 2026)
Save grid                         yearly                               every 0.1 year plus the screening
                                                                       ramp times, so the trapezoid
                                                                       integral is accurate
Solver                            Dormand–Prince, rtol=atol=1e-10      Tsit5, PID controller at
                                  (the goldens' setting)               rtol=1e-6, atol=1e-9, told about
                                                                       the yearly mixing-matrix jumps
                                                                       and the screening ramps
================================  ===================================  ==================================

``docs/08-parity-and-improved.md`` and ``scripts/compare_improved.py`` show the numerical
differences and why the improved choices are closer to the quantities the outputs stand for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import diffrax
import numpy as np

from summer4.solvers import Diffrax

START_TIME = 1850.0
END_TIME = 2035.0
# The age-mixing matrix is rebuilt each calendar year, so the vector field jumps at every integer
# year. Telling the step-size controller (``jump_ts``) lets it end a step on each jump instead of
# rejecting steps that straddle one.
MIXING_JUMPS: np.ndarray = np.arange(START_TIME + 1.0, END_TIME)


def _parity_solver(jumps: np.ndarray) -> dict[str, Any]:
    # summer2gen's controller knows nothing about jumps; neither does the parity solve.
    return {"solver": "dopri5", "rtol": 1e-10, "atol": 1e-10}


def _improved_solver(jumps: np.ndarray) -> dict[str, Any]:
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-9, jump_ts=jumps)
    return {"solver": Diffrax(diffrax.Tsit5(), stepsize_controller=controller), "max_steps": 16384}


@dataclass(frozen=True)
class Variant:
    """How to solve the model and report its flow outputs.

    Attributes:
        name: ``"parity"`` or ``"improved"``.
        flow_values: ``OutputSet(flow_values=...)``: ``"midpoint"`` or ``"rate"``.
        cumulative: ``"cumsum"`` (amounts per interval) or ``"cumulative_integral"`` (a rate).
        save_step: Spacing of the save grid, in years.
        solver: Builds the ``run`` solver keywords from the times where the vector field jumps
            or has a kink.
        use_breakpoints: Whether the model's breakpoints (the screening ramps) are added to the
            save grid and given to the solver.
    """

    name: str
    flow_values: Literal["midpoint", "rate"]
    cumulative: Literal["cumsum", "cumulative_integral"]
    save_step: float
    solver: Callable[[np.ndarray], dict[str, Any]] = field(repr=False)
    use_breakpoints: bool = False

    def save_times(self, breakpoints: Sequence[float] = ()) -> np.ndarray:
        """The save grid from 1850 to 2035, plus ``breakpoints`` if the variant uses them."""
        n = int(round((END_TIME - START_TIME) / self.save_step))
        grid = np.linspace(START_TIME, END_TIME, n + 1)
        if not self.use_breakpoints or not len(breakpoints):
            return grid
        return np.unique(np.round(np.concatenate([grid, np.asarray(breakpoints)]), 9))

    def run_kwargs(self, breakpoints: Sequence[float] = ()) -> dict[str, Any]:
        """Solver keywords for ``CompiledModel.run``."""
        jumps = MIXING_JUMPS
        if self.use_breakpoints and len(breakpoints):
            jumps = np.unique(np.concatenate([MIXING_JUMPS, np.asarray(breakpoints)]))
        return self.solver(jumps)

    def with_save_step(self, save_step: float) -> Variant:
        """The same variant on another save grid."""
        return replace(self, name=f"{self.name}@{save_step:g}", save_step=save_step)


# The goldens were made with summer2gen's Dormand–Prince ``odeint`` at rtol = atol = 1e-10; this
# is the same method and tolerance in diffrax. At 1e-8 both solvers carry about 1.5e-5 relative
# error from the yearly mixing jumps, above the 1e-5 parity bar.
PARITY = Variant(
    name="parity",
    flow_values="midpoint",
    cumulative="cumsum",
    save_step=1.0,
    solver=_parity_solver,
)

IMPROVED = Variant(
    name="improved",
    flow_values="rate",
    cumulative="cumulative_integral",
    save_step=0.1,
    solver=_improved_solver,
    use_breakpoints=True,
)

VARIANTS: dict[str, Variant] = {v.name: v for v in (PARITY, IMPROVED)}
