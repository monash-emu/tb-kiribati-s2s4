---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The same model, the summer4 way

`kiribati_tb/model.py` follows `tbh/model.py` function for function: build the natural
history, stratify, then adjust flows by name. That is the right first port, because it can be
read beside the original. Once it passes the goldens, summer4 allows a much more compact model.
`kiribati_tb/model_summer4.py` builds the **same model** that way, with the same `get_tb_model`
call and result, and passes the same golden tests.

## What changes

**The map first.** `STATE × AGE × REACH` is declared up front, so each flow is written once, with
all its adjustments, instead of being declared, stratified and then adjusted by name in three
places.

**One flow over a selector.** Where the original has one flow per compartment with the same
rate, or rates that differ by a per-compartment factor, the compact model has one flow whose
source is a selector, and the factor is a `Multiply(..., where=STATE[...])`:

::::{grid} 1 1 2 2
:::{grid-item-card} `model.py`: four flows
```python
for susceptible_comp in ["mtb_naive", "contained",
                         "cleared", "recovered"]:
    if susceptible_comp == "recovered":
        rel_susceptibility = Param("rel_sus_cleared")
    else:
        rel_susceptibility = (
            1.0 if susceptible_comp == "mtb_naive"
            else Param(f"rel_sus_{susceptible_comp}"))
    model.add_flow(TransitionFlow(
        f"infection_from_{susceptible_comp}",
        STATE[susceptible_comp], STATE["incipient"], foi,
        adjust=(Multiply(rel_susceptibility),)))
...  # in stratify_model_by_age / _by_reachability:
model.adjust_flow("infection_from_mtb_naive",
    Multiply(Param("rel_sus_children"), where=CHILDREN))
for flow in infection_flows:
    model.adjust_flow(flow, Multiply(
        Param("rel_sus_unreachable"), where=unreachable))
```
:::
:::{grid-item-card} `model_summer4.py`: one flow
```python
TransitionFlow(
    "infection",
    s.isin(["mtb_naive", "contained",
            "cleared", "recovered"]),
    s["incipient"],
    foi,
    adjust=[
        Multiply(Param("rel_sus_children"),
                 where=s["mtb_naive"] & CHILDREN),
        Multiply(Param("rel_sus_contained"),
                 where=s["contained"]),
        Multiply(Param("rel_sus_cleared"),
                 where=s.isin(["cleared", "recovered"])),
        Multiply(Param("rel_sus_unreachable"),
                 where=UNREACHABLE),
    ],
)
```
:::
::::

The same goes for passive detection (4 flows → 1), self-recovery (2 → 1), TB mortality (2 → 1),
each screening program (one flow per detected state → 1, the sensitivity per state) and births
(2 → 1, an `EntryFlow` with a parameter `split` over reachability).

**`TraitChain` within the state property.** Clinical progression moves `subclin_inf → clin_inf`
and `subclin_lowinf → clin_lowinf` at one rate. Those are explicit pairs of traits of one
property, which is exactly what `TraitChain` describes; ageing uses it over `AGE`, and it works
the same over `STATE`:

::::{grid} 1 1 2 2
:::{grid-item-card} `model.py`: two flows per move, four moves
```python
for infectious_status in ["inf", "lowinf"]:
    model.add_flow(TransitionFlow(
        f"clinical_progression_{infectious_status}",
        STATE[f"subclin_{infectious_status}"],
        STATE[f"clin_{infectious_status}"],
        Param("clinical_progression_rate")))
    model.add_flow(TransitionFlow(
        f"clinical_regression_{infectious_status}",
        STATE[f"clin_{infectious_status}"],
        STATE[f"subclin_{infectious_status}"],
        Param("clinical_regression_rate")))
# ... and the same for infectiousness gain and loss
```
:::
:::{grid-item-card} `model_summer4.py`: one flow per move
```python
CLINICAL = (("subclin_inf", "clin_inf"),
            ("subclin_lowinf", "clin_lowinf"))
INFECTIOUSNESS = (("subclin_lowinf", "subclin_inf"),
                  ("clin_lowinf", "clin_inf"))

def chain(name, pairs, rate, **kw):
    return TransitionFlow(
        name, STATE.present(), STATE.present(), rate,
        pairing=TraitChain(STATE, pairs), **kw)

chain("clinical_progression", CLINICAL, Param("clinical_progression_rate"))
chain("clinical_regression", reverse(CLINICAL), Param("clinical_regression_rate"))
chain("infectiousness_gain", INFECTIOUSNESS, Param("infectiousness_gain_rate"))
chain("infectiousness_loss", reverse(INFECTIOUSNESS), Param("infectiousness_loss_rate"))
```
:::
::::

Progression from `incipient` to the two subclinical states is a chain too: two pairs with the
same source, and the infectious share as per-pair rates, `rates=(1 - p, p)`. A rate can be a
number (an ageing band's `1/5`) or any rate expression (summer4 PR #84 on; before that the port
wrote `Multiply(p, where=Dest(s["subclin_inf"]))` and `Multiply(1 - p, where=Dest(s["subclin_lowinf"]))`,
which is what `rates=` compiles to).

## The edges of a chain

`cm.edges(name)` lists the compartment-to-compartment edges a flow resolved to. Each chain should
connect exactly its state pairs, once in each of the 16 age × reachability strata, without
changing age or reachability:

```{code-cell} ipython3
import numpy as np
import pandas as pd

from kiribati_tb import model, model_summer4
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors
from kiribati_tb.scenarios import SCENARIOS_BY_ID

_, _, tv_params = get_parameters_and_priors()
programs = SCENARIOS_BY_ID["scenario_3"].scr_prgs
compact = model_summer4.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params, programs)
familiar = model.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params, programs)


def edges(cm, name):
    e = cm.edges(name)
    rows = e.pmap.to_dicts()
    return pd.DataFrame(
        [{"from": rows[s]["state"], "to": rows[d]["state"],
          "age": rows[s]["age"] + ("" if rows[s]["age"] == rows[d]["age"] else " → " + rows[d]["age"]),
          "reach": rows[s]["reach"] + ("" if rows[s]["reach"] == rows[d]["reach"] else " → ?")}
         for s, d in zip(np.asarray(e.src_idx), np.asarray(e.dest_idx))])


gain = edges(compact.model, "infectiousness_gain")
print(len(gain), "edges")
gain.head(6)
```

```{code-cell} ipython3
summary = {
    name: edges(compact.model, name).groupby(["from", "to"]).size()
    for name in ["clinical_progression", "clinical_regression", "infectiousness_gain",
                 "infectiousness_loss", "progression"]
}
pd.concat(summary, names=["flow"]).rename("edges (one per age × reach stratum)").to_frame()
```

Every pair appears 16 times, and age and reachability never change along an edge.
`tests/test_model_summer4.py` asserts this for every chain.

## Why the trajectories are the same

Each merged flow has, on every edge, exactly the rate the original flow had on that edge: the
same base rate times the same factors, only selected by `where=` on one flow instead of being
spread over several. So the two models have the same vector field. The cell below evaluates both
at the 1850 state and at a state from 2025, and compares:

```{code-cell} ipython3
from kiribati_tb.runner_tools import midpoint_params, run_compartments
from kiribati_tb.variants import PARITY

params = midpoint_params()
assert familiar.model.pmap.labels() == compact.model.pmap.labels()
y1850 = np.asarray(familiar.model.initial_state(familiar.model.prepare(params)).data)
y2025 = run_compartments(familiar, params, PARITY).loc[2025.0].to_numpy()
rows = {}
for label, (t, y) in {"1850": (1850.0, y1850), "2025": (2025.0, y2025)}.items():
    da = np.asarray(familiar.model.vector_field(t, y, params))
    db = np.asarray(compact.model.vector_field(t, y, params))
    rows[label] = np.max(np.abs(da - db)) / np.max(np.abs(da))
pd.Series(rows, name="max |dy/dt difference| / max |dy/dt|")
```

They agree to floating-point rounding. Merging flows changes only the order of some
additions, and an adaptive solver at `rtol = 1e-10` then picks marginally different steps. So
the two trajectories differ by about the solver's own error (1e-7 relative), and both pass the
goldens at the same tolerances: `tests/test_parity.py` runs every golden test on both builders.

## What outputs need

A flow that no longer exists can't be named in an output, but its original name can be rebuilt by
selecting edges of the flow that now carries it. `model_summer4.flow_sources` maps each original
flow name to `(flow, source, dest)`:

| Original flow | Now |
| --- | --- |
| `tb_detection_clin_inf` | `("tb_detection", STATE["clin_inf"], None)` |
| `progression_inf` | `("progression", None, STATE["subclin_inf"])` |
| `births_reachable` | `("births", None, REACH["reachable"])` |
| `pearl_10+_clin_inf` | `("pearl_10+", STATE["clin_inf"], None)` |

`request_model_outputs(..., flow_sources=...)` resolves them, so `notifications_clin_inf` is
`outputs.flow("notifications_clin_inf", "tb_detection", source=STATE["clin_inf"])`, and all ~680
output names and values are unchanged. The output module is shared by both models.

## Counts

```{code-cell} ipython3
from pathlib import Path


def code_lines(path):
    import ast
    src = Path(path).read_text()
    tree, doc = ast.parse(src), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            doc.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    lines = src.splitlines()
    return len(lines), sum(1 for i, l in enumerate(lines, 1)
                           if l.strip() and not l.strip().startswith("#") and i not in doc)


baseline = {
    name: get(DEFAULT_MODEL_CONFIG, tv_params)
    for name, get in [("model.py", model.get_tb_model), ("model_summer4.py", model_summer4.get_tb_model)]
}
pd.DataFrame({
    "lines": [code_lines(f"../src/kiribati_tb/{n}")[0] for n in baseline],
    "code lines": [code_lines(f"../src/kiribati_tb/{n}")[1] for n in baseline],
    "flows (baseline)": [len(tb.model.flows) for tb in baseline.values()],
    "flows (scenario 3)": [len(familiar.model.flows), len(compact.model.flows)],
}, index=list(baseline))
```

`model_summer4.py` reuses the constants, the force of infection, the treatment-outcome function
and the scaleup from `model.py`. Counting those, it is still about two thirds the size. It has
about half the flows.

## Going further

The four disease states are really two axes. [TB disease as two properties](14-disease-strata)
makes each a property, so the paired-state chains here become single-axis flows.

## Which to use

Port with the `model.py` shape: it is the one a summer2 user can check line by line against the
original, and the goldens find any mistake. Once it passes, the compact form is easier to read
and change on its own terms: a new active state is one more pair in a chain, not two more flows
and four more adjustments. Keep both passing the same goldens, as this repository does.
