"""The calibration model builds with the stock ``TargetSet`` and scores every target."""

from __future__ import annotations

import jax
import numpy as np

from kiribati_tb.calibration import get_bcm, get_targets
from kiribati_tb.runner_tools import midpoint_params


def test_targets_name_model_outputs() -> None:
    bcm, tb = get_bcm()
    names = set(tb.outputs.keys())
    assert {t.key for t in get_targets()} <= names
    assert len(bcm.priors) >= 15


def test_log_likelihood_and_gradient_are_finite() -> None:
    bcm, _ = get_bcm()
    params = midpoint_params()
    draws = {p.name: params[p.name] for p in bcm.priors} | {"mixing_dist_sd": 10.0}
    value = bcm.log_likelihood(draws)
    assert np.isfinite(float(value))
    grad = jax.grad(lambda d: bcm.log_likelihood(d))(
        {k: jax.numpy.asarray(v) for k, v in draws.items()}
    )
    assert all(np.isfinite(float(g)) for g in grad.values())
