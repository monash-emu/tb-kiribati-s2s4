"""Structure of the compact model: TraitChain within the state property, and merged flows."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from kiribati_tb import model, model_summer4
from kiribati_tb.model import AGE, REACH, TBModel
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors
from kiribati_tb.scenarios import SCENARIOS_BY_ID

CHAINS = {
    "clinical_progression": model_summer4.CLINICAL,
    "clinical_regression": model_summer4.reverse(model_summer4.CLINICAL),
    "infectiousness_gain": model_summer4.INFECTIOUSNESS,
    "infectiousness_loss": model_summer4.reverse(model_summer4.INFECTIOUSNESS),
    "progression": model_summer4.PROGRESSION,
}


@cache
def built(builder: str, scenario: str | None = "scenario_3") -> TBModel:
    _, _, tv_params = get_parameters_and_priors()
    get = model.get_tb_model if builder == "tbh_style" else model_summer4.get_tb_model
    return get(DEFAULT_MODEL_CONFIG, tv_params, SCENARIOS_BY_ID[scenario].scr_prgs)


def edge_rows(name: str) -> list[dict[str, str]]:
    edges = built("summer4_way").model.edges(name)
    rows = edges.pmap.to_dicts()
    return [
        {"src_" + k: v for k, v in rows[s].items()} | {"dest_" + k: v for k, v in rows[d].items()}
        for s, d in zip(np.asarray(edges.src_idx), np.asarray(edges.dest_idx))
    ]


@pytest.mark.parametrize("name", CHAINS)
def test_trait_chain_connects_exactly_the_state_pairs_in_every_stratum(name: str) -> None:
    rows = edge_rows(name)
    strata = [(a, r) for a in AGE.traits for r in REACH.traits]
    expected = {(src, dest, a, r) for src, dest in CHAINS[name] for a, r in strata}
    got = {(e["src_state"], e["dest_state"], e["src_age"], e["src_reach"]) for e in rows}
    assert got == expected
    assert len(rows) == len(expected)  # no duplicate edges
    assert all(e["src_age"] == e["dest_age"] and e["src_reach"] == e["dest_reach"] for e in rows)


def test_fewer_flows_than_the_tbh_style_model() -> None:
    compact = set(built("summer4_way").model.flows)
    familiar = set(built("tbh_style").model.flows)
    assert len(compact) < len(familiar)
    assert {"infection", "tb_detection", "self_recovery", "tb_mortality", "births"} <= compact
    assert all(f"infection_from_{s}" in familiar for s in ("mtb_naive", "recovered"))


def test_every_original_flow_name_is_kept_or_mapped() -> None:
    """Names gone from the compact model are in ``flow_sources``, so outputs can select them."""
    familiar = set(built("tbh_style").model.flows)
    compact = set(built("summer4_way").model.flows)
    mapped = set(model_summer4.flow_sources(SCENARIOS_BY_ID["scenario_3"].scr_prgs))
    read_by_outputs = {
        n for n in familiar - compact if not n.startswith(("infection_from", "self_recovery"))
    }
    leftover = read_by_outputs - mapped
    assert leftover <= set(CHAINS) | {
        "clinical_progression_inf",
        "clinical_progression_lowinf",
        "clinical_regression_inf",
        "clinical_regression_lowinf",
        "infectiousnnes_gain_clin",
        "infectiousnnes_gain_subclin",
        "infectiousnnes_loss_clin",
        "infectiousnnes_loss_subclin",
        "tb_mortality_inf",
        "tb_mortality_lowinf",
    }, leftover
