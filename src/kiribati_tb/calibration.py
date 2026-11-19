"""Bayesian calibration on numpyro through summer4's ``BayesianModel``.

The original calibrates with estival (``BayesianCompartmentalModel``) and pymc. The posterior
is the same here: priors from the ``parameters.xlsx`` constant sheet, the original's ten
targets with its tolerance rule, and the mixing-matrix distance target with a prior on its
standard deviation. Every target names a derived output of the model's ``OutputSet``
(``..._per100k``, ``..._perc``, ``notifications``, ``mixing_matrix_distance``), and
``BayesianModel(outputs=...)`` scores them on the evaluated outputs directly.

The calibration solve is the original's: summer2gen's Dormand–Prince at its default
``rtol = atol = 1.4e-4`` on the parity outputs. ``CALIBRATION_IMPROVED`` uses the improved
variant's solver instead (more accurate for fewer steps; see ``docs/06``).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from summer4 import Target, TargetSet
from summer4.epi.calibration import BayesianModel, NormalLikelihood, Uniform, priors_from_frame
from summer4.epi.calibration.priors import Prior

from kiribati_tb.model import TBModel, get_tb_model
from kiribati_tb.paths import DATA_FOLDER
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors
from kiribati_tb.variants import END_TIME, IMPROVED, PARITY, START_TIME, Variant

MIXING_PRIORS = ("bg_mixing", "a_spread", "pc_strength")
CALIBRATION_SOLVER: dict[str, Any] = {
    "solver": "dopri5",
    "rtol": 1.4e-4,
    "atol": 1.4e-4,
    "max_steps": 4096,
}


@cache
def read_notifications() -> dict[float, float]:
    """Historical notifications for the modelled population, before 2023."""
    frame = pd.read_excel(DATA_FOLDER / "notifications.xlsx")
    frame = frame[frame["year"] < 2023]
    return {float(y): float(v) for y, v in zip(frame["year"], frame["inputed_all_other"])}


def normal_target(
    output: str, observations: Mapping[float, float], tol_pct: float = 20.0
) -> Target:
    """estival's ``NormalTarget`` with the original's tolerance rule (``get_normal_target``)."""
    years = np.asarray(sorted(observations), dtype=float)
    values = np.asarray([observations[y] for y in years], dtype=float)
    likelihood = NormalLikelihood.from_tolerance(values, tol_pct, aggregate="mean")
    return Target(key=output, times=years, values=values, likelihood=likelihood)


def get_targets(heterogeneous_mixing: bool = True) -> list[Target]:
    """The original's calibration targets."""
    targets = [
        normal_target("pearl_posXreach_reachable_per100k", {2024.0: 850.2}),
        normal_target("cxr_posXreach_reachable_per100k", {2024.0: 595.9}),
        normal_target("tst_posXage_3_9Xreach_reachable_perc", {2024.0: 3.3}),
        normal_target("tst_posXage_10Xreach_reachable_perc", {2024.0: 9.3}),
        normal_target("tst_posXage_15+Xreach_reachable_perc", {2024.0: 28.23}),
        normal_target("tst_posXage_18+Xreach_reachable_perc", {2011.0: 38.0}),
        normal_target("perc_prev_subclinicalXreach_reachable", {2024.0: 81.6}),  # 115/(115+26)
        normal_target("perc_prev_infectiousXreach_reachable", {2024.0: 69.9}),  # 100/143
        normal_target("notifications", read_notifications(), tol_pct=40.0),
    ]
    if heterogeneous_mixing:
        # Distance to the conmat matrix in 2025 ~ Normal(0, sd), sd ~ Uniform(5, 20).
        targets.append(
            Target(
                key="mixing_matrix_distance",
                times=np.asarray([2025.0]),
                values=np.asarray([0.0]),
                likelihood=NormalLikelihood(sd=Uniform("mixing_dist_sd", 5.0, 20.0)),
            )
        )
    return targets


def get_priors(params: Mapping[str, float], heterogeneous_mixing: bool = True) -> list[Prior]:
    """Priors from the constant sheet, with the original's ``infectiousness_gain_rate`` rule.

    When ``infectiousness_loss_rate > 3`` the original narrows ``infectiousness_gain_rate`` to
    Uniform(2, 10).
    """
    _, prior_rows, _ = get_parameters_and_priors()
    priors = list(
        priors_from_frame(prior_rows, "parameter", "distribution", "distri_param1", "distri_param2")
    )
    if params["infectiousness_loss_rate"] > 3.0:
        priors = [p for p in priors if p.name != "infectiousness_gain_rate"]
        priors.append(Uniform("infectiousness_gain_rate", 2.0, 10.0))
    if not heterogeneous_mixing:
        priors = [p for p in priors if p.name not in MIXING_PRIORS]
    return priors


def get_bcm(
    params: Mapping[str, float] | None = None,
    heterogeneous_mixing: bool = True,
    variant: Variant = PARITY,
    run_kwargs: Mapping[str, Any] | None = None,
) -> tuple[BayesianModel, TBModel]:
    """The calibration model (estival's ``get_bcm``), and the ``TBModel`` it wraps.

    Args:
        params: Fixed parameter values; the constant sheet by default.
        heterogeneous_mixing: ``False`` for the homogeneous-mixing sensitivity analysis.
        variant: Output convention of the scored outputs (``PARITY`` as the original).
        run_kwargs: Solver keywords; the original's calibration solver by default.
    """
    sheet_params, _, tv_params = get_parameters_and_priors()
    params = dict(sheet_params if params is None else params)
    config = DEFAULT_MODEL_CONFIG | {"heterogeneous_mixing": heterogeneous_mixing}
    tb = get_tb_model(config, tv_params, (), variant)
    targets = get_targets(heterogeneous_mixing)
    # Score on the targets' outputs only; the rest of the OutputSet is not evaluated.
    outputs = tb.outputs.subset([t.key for t in targets])
    kwargs = {
        "t0": START_TIME,
        "t1": END_TIME,
        "dt": 1.0,
        **(CALIBRATION_SOLVER if run_kwargs is None else run_kwargs),
    }
    bcm = BayesianModel(
        tb.model,
        params,
        get_priors(params, heterogeneous_mixing),
        TargetSet(tuple(targets)),
        outputs=outputs,
        run_kwargs=kwargs,
    )
    return bcm, tb


CALIBRATION_IMPROVED: dict[str, Any] = IMPROVED.run_kwargs()
