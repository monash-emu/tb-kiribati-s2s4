"""Time-varying inputs and the mixing matrix equal the original summer2 functions.

``demography.parquet`` holds the original's per-age death rates, births entry rate, treatment
success, negative treatment outcomes and passive detection rate every half year from 1935 to
2035; ``mixing.parquet`` its mixing matrix for five parameter draws at ten times. The port's rate
expressions are saved during a solve with ``outputs[name] = expression``: the tool summer4 now
has for exactly this.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from summer4 import OutputSet, Param, SavePlan, Time
from summer4.data import Data

from kiribati_tb.age_mixing import (
    START_YEAR,
    YEARS,
    canberra_distance,
    contact_rates,
    default_mixing_inputs,
    read_conmat_matrix,
    spectral_radius,
    yearly_matrices,
)
from kiribati_tb.demographic_tools import AGE_GROUPS, load_demography
from kiribati_tb.model import (
    AGE,
    SUMMER2_SIGMOID_CURVATURE,
    get_neg_tx_outcome_funcs,
    get_tb_model,
    tanh_based_scaleup,
)
from kiribati_tb.paths import GOLDEN_FOLDER
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors
from parity import read_golden, read_params

RTOL = 1e-9
N_BANDS = 8


@pytest.fixture(scope="module")
def demography_golden() -> pd.DataFrame:
    return pd.read_parquet(GOLDEN_FOLDER / "demography.parquet")


@pytest.fixture(scope="module")
def saved_inputs(demography_golden: pd.DataFrame) -> dict[str, np.ndarray]:
    """The port's input expressions, saved at the golden times during a model solve."""
    _, _, tv_params = get_parameters_and_priors()
    demog = load_demography()
    death = demog.death_rate_table(AGE).interp("sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE)
    tsr = tv_params["tx_success_pct"]
    success = Data(tsr.index.to_numpy(float), tsr.to_numpy(float) / 100.0).interp("linear")
    outcomes = get_neg_tx_outcome_funcs(death, success)
    entry = Data(demog.entry_times, demog.entry_values).interp(
        "sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE
    )
    detection = Param("recent_detection_rate") * tanh_based_scaleup(
        Time(),
        Param("passive_detection_shape"),
        Param("passive_detection_inflection"),
        Param("passive_detection_past_frac"),
    )
    outputs = OutputSet()
    for name, expression in {
        "death_rate": death,
        "tx_relapse": outcomes.relapse,
        "tx_death": outcomes.death,
        "entry_rate": entry,
        "tx_success": success,
        "passive_detection_rate": detection,
    }.items():
        outputs[name] = expression
    tb = get_tb_model(DEFAULT_MODEL_CONFIG | {"heterogeneous_mixing": False}, tv_params)
    times = demography_golden["time"].to_numpy()
    params = read_params("hetero_baseline")
    result = tb.model.run(
        params, t0=1850.0, t1=2035.0, dt=0.5, save=outputs.plan(SavePlan(ts=times)), solver="euler"
    )
    named = outputs.evaluate(result, params)
    return {name: np.asarray(named[name].array()) for name in named.keys()}


@pytest.mark.parametrize("name", ["death_rate", "tx_relapse", "tx_death"])
def test_per_age_series(
    name: str, demography_golden: pd.DataFrame, saved_inputs: dict[str, np.ndarray]
) -> None:
    expected = demography_golden[[f"{name}_{age}" for age in AGE_GROUPS]].to_numpy()
    np.testing.assert_allclose(saved_inputs[name], expected, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", ["entry_rate", "tx_success", "passive_detection_rate"])
def test_scalar_series(
    name: str, demography_golden: pd.DataFrame, saved_inputs: dict[str, np.ndarray]
) -> None:
    got = saved_inputs[name]
    np.testing.assert_allclose(got, demography_golden[name].to_numpy(), rtol=RTOL, atol=1e-12)


@pytest.fixture(scope="module")
def mixing_golden() -> pd.DataFrame:
    return pd.read_parquet(GOLDEN_FOLDER / "mixing.parquet")


def _mixing(bg: float, spread: float, pc: float) -> dict[str, float]:
    return {"bg_mixing": bg, "a_spread": spread, "pc_strength": pc}


def test_stack_equals_original_builder(mixing_golden: pd.DataFrame) -> None:
    """Five parameter draws x ten times, including within-year times and the 1950 data start."""
    columns = [f"m_{i}_{j}" for i in range(N_BANDS) for j in range(N_BANDS)]
    for (bg, spread, pc), rows in mixing_golden.groupby(["bg_mixing", "a_spread", "pc_strength"]):
        stack = np.asarray(yearly_matrices(_mixing(bg, spread, pc)))
        years = np.floor(rows["time"].to_numpy()).astype(int) - START_YEAR
        expected = rows[columns].to_numpy().reshape(-1, N_BANDS, N_BANDS)
        np.testing.assert_allclose(stack[years], expected, rtol=1e-12, atol=1e-15)


def test_eigvalsh_radius_equals_eigvals_radius() -> None:
    """``rho(S diag N)`` from the symmetric form equals the general eigenvalue routine."""
    inputs = default_mixing_inputs()
    rng = np.random.default_rng(20260929)
    populations = jnp.asarray(inputs.populations)
    for _ in range(20):
        params = _mixing(rng.uniform(0.01, 0.05), rng.uniform(2.0, 15.0), rng.uniform(0.01, 1.0))
        contacts = contact_rates(params, inputs)
        general = np.max(
            np.abs(np.linalg.eigvals(np.asarray(contacts * populations[:, None, :]))), axis=-1
        )
        np.testing.assert_allclose(
            np.asarray(spectral_radius(contacts, populations)), general, rtol=1e-12
        )


def test_stack_is_differentiable_in_every_mixing_parameter() -> None:
    def total(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(
            yearly_matrices(dict(zip(("bg_mixing", "a_spread", "pc_strength"), values))) ** 2
        )

    grad = np.asarray(jax.grad(total)(jnp.asarray([0.03, 8.5, 0.5])))
    assert np.all(np.isfinite(grad)) and np.all(grad != 0.0)


def test_mixing_stack_is_hoisted_to_run_start() -> None:
    """The parameter-only ``defer`` that builds the yearly stack is computed once per run."""
    _, _, tv = get_parameters_and_priors()
    model = get_tb_model(DEFAULT_MODEL_CONFIG, tv).model
    hoisted = [repr(entry) for entry in model.hoist_table.entries]
    assert any("build_mixing_stack" in text for text in hoisted), hoisted


def test_distance_to_conmat_by_year() -> None:
    """Canberra distance to the conmat matrix, per year, as the original computed value."""
    stack = yearly_matrices(_mixing(0.03, 8.5, 0.505))
    distance = np.asarray(canberra_distance(stack, read_conmat_matrix()))
    golden = read_golden("hetero_baseline", "derived")
    np.testing.assert_allclose(
        distance, golden.loc[YEARS.astype(float), "mixing_matrix_distance"], rtol=1e-12
    )
