"""Golden parity: the ``PARITY`` variant reproduces the original summer2gen model.

Every compartment and every derived output the original computed (including its
``save_results=False`` intermediates) is compared, by the original's names, for five fixtures:
no transmission, homogeneous and heterogeneous mixing, and two screening scenarios. The goldens
and the comparison helpers are the ones ``kiribati-tb-summer4`` uses.
"""

from __future__ import annotations

from functools import cache

import pandas as pd
import pytest

from kiribati_tb import model, model_summer4
from kiribati_tb.model import TBModel
from kiribati_tb.runner_tools import (
    DEFAULT_MODEL_CONFIG,
    compartment_labels,
    get_parameters_and_priors,
    run_compartments,
    run_model,
)
from kiribati_tb.scenarios import SCENARIOS, SCENARIOS_BY_ID
from kiribati_tb.variants import PARITY
from parity import assert_parity, read_golden, read_params

FIXTURES: dict[str, tuple[bool, str | None]] = {
    # fixture: (heterogeneous mixing, scenario)
    "no_transmission": (False, None),
    "homog_baseline": (False, None),
    "hetero_baseline": (True, None),
    "hetero_scenario_3": (True, "scenario_3"),
    "hetero_scenario_8": (True, "scenario_8"),
}
# The tbh-style model (kiribati_tb.model) and the compact one (kiribati_tb.model_summer4) build
# the same model; every golden test runs on both.
BUILDERS = {"tbh_style": model.get_tb_model, "summer4_way": model_summer4.get_tb_model}
CASES = [(fixture, builder) for builder in BUILDERS for fixture in FIXTURES]
ATOL_COMPARTMENTS = 1e-6  # persons
ATOL_OUTPUTS = 1e-8
# Without transmission the disease states are solver noise (~1e-10 persons, and negative) once
# the seed dies out; per-capita outputs of them are compared at the compartment floor instead.
ATOL_NO_TRANSMISSION = 1e-6


@cache
def tb_model(fixture: str, builder: str = "tbh_style") -> TBModel:
    hetero, scenario = FIXTURES[fixture]
    _, _, tv_params = get_parameters_and_priors()
    config = DEFAULT_MODEL_CONFIG | {"heterogeneous_mixing": hetero}
    programs = SCENARIOS_BY_ID[scenario].scr_prgs if scenario else []
    return BUILDERS[builder](config, tv_params, programs, PARITY)


@cache
def derived(fixture: str, builder: str) -> pd.DataFrame:
    built = tb_model(fixture, builder)
    return run_model(built, read_params(fixture), PARITY, names=list(built.outputs.keys()))


def test_scenario_ids_and_names_are_the_originals() -> None:
    assert [s.sc_id for s in SCENARIOS] == [f"scenario_{i}" for i in (1, 2, 3, 6, 7, 8, 16, 17, 18)]
    assert SCENARIOS_BY_ID["scenario_3"].sc_name == "3. PEARL / Vhigh"
    assert SCENARIOS_BY_ID["scenario_8"].sc_name == "8. CXR-TST / 84%"
    assert SCENARIOS_BY_ID["scenario_17"].sc_name_2 == "Drop Xpert & TST | 75%"


@pytest.mark.parametrize(("fixture", "builder"), CASES)
def test_compartment_names_are_the_originals(fixture: str, builder: str) -> None:
    labels = compartment_labels(tb_model(fixture, builder))
    assert len(labels) == 160
    assert set(labels) == set(read_golden(fixture, "compartments").columns)


@pytest.mark.parametrize(("fixture", "builder"), CASES)
def test_compartments_match_golden(fixture: str, builder: str) -> None:
    got = run_compartments(tb_model(fixture, builder), read_params(fixture), PARITY)
    golden = read_golden(fixture, "compartments")
    assert (got.loc[1850.0, golden.columns] == golden.loc[1850.0]).all(), "initial population"
    report = assert_parity(got, golden, ATOL_COMPARTMENTS)
    print(fixture, builder, report.head(5).to_string(), sep="\n")


@pytest.mark.parametrize(("fixture", "builder"), CASES)
def test_output_names_are_the_originals(fixture: str, builder: str) -> None:
    """Every name the original computed, and no others (``by=`` groupings aside)."""
    names = {n for n in tb_model(fixture, builder).outputs.keys() if "[" not in n}
    assert names == set(read_golden(fixture, "derived").columns)


@pytest.mark.parametrize(("fixture", "builder"), CASES)
def test_every_output_matches_golden(fixture: str, builder: str) -> None:
    frame = derived(fixture, builder)
    golden = read_golden(fixture, "derived")
    if fixture == "no_transmission":
        # Once the seed has died out, ratios of disease states (% subclinical) are ratios of
        # solver noise in both models. Compare them only while there is disease to divide.
        noise = [c for c in golden.columns if c.startswith(("perc_prev_", "perc_notif"))]
        early = golden["tb_prevalence"] > 1e-3
        assert_parity(frame.loc[early, noise], golden.loc[early, noise], ATOL_OUTPUTS)
        golden = golden.drop(columns=noise)
    atol = ATOL_NO_TRANSMISSION if fixture == "no_transmission" else ATOL_OUTPUTS
    report = assert_parity(frame, golden, atol)
    print(fixture, builder, report.head(5).to_string(), sep="\n")
