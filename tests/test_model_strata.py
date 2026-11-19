"""Structure of the disease-strata model: one axis moves per flow, the other is carried along."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from kiribati_tb import model_strata
from kiribati_tb.model import AGE, REACH, TBModel
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, compartment_labels
from kiribati_tb.runner_tools import get_parameters_and_priors
from kiribati_tb.scenarios import SCENARIOS_BY_ID

# flow: (source cell, dest cell) for each edge's (state, clin, inf), as summer2 state names
EXPECTED = {
    "clinical_progression": {("subclin_inf", "clin_inf"), ("subclin_lowinf", "clin_lowinf")},
    "clinical_regression": {("clin_inf", "subclin_inf"), ("clin_lowinf", "subclin_lowinf")},
    "infectiousness_gain": {("subclin_lowinf", "subclin_inf"), ("clin_lowinf", "clin_inf")},
    "infectiousness_loss": {("subclin_inf", "subclin_lowinf"), ("clin_inf", "clin_lowinf")},
    "progression": {("incipient", "subclin_lowinf"), ("incipient", "subclin_inf")},
    "self_recovery": {("subclin_lowinf", "recovered"), ("subclin_inf", "recovered")},
    "tx_relapse": {("treatment", "subclin_lowinf")},
    "tb_mortality": {("clin_lowinf", "mtb_naive"), ("clin_inf", "mtb_naive")},
}


@cache
def built() -> TBModel:
    _, _, tv_params = get_parameters_and_priors()
    programs = SCENARIOS_BY_ID["scenario_3"].scr_prgs
    return model_strata.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params, programs)


def state_name(row: dict[str, str]) -> str:
    return f"{row['clin']}_{row['inf']}" if "clin" in row else row["state"]


@pytest.mark.parametrize("name", EXPECTED)
def test_flow_connects_the_intended_cells_in_every_stratum(name: str) -> None:
    edges = built().model.edges(name)
    rows = edges.pmap.to_dicts()
    pairs = [
        (rows[s], rows[d]) for s, d in zip(np.asarray(edges.src_idx), np.asarray(edges.dest_idx))
    ]
    assert {(state_name(a), state_name(b)) for a, b in pairs} == EXPECTED[name]
    assert all(a["reach"] == b["reach"] for a, b in pairs)
    if name != "tb_mortality":  # deaths return to the youngest age
        assert all(a["age"] == b["age"] for a, b in pairs)
    per_cell = len(AGE.traits) * len(REACH.traits)
    assert len(pairs) == len(EXPECTED[name]) * per_cell


def test_same_compartments_as_the_original() -> None:
    labels = compartment_labels(built())
    assert len(labels) == len(set(labels)) == 160
    assert "subclin_lowinfXage_15Xreachability_reachable" in labels
