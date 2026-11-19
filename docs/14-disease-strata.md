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

# TB disease as two properties

The original has four disease states, `subclin_lowinf`, `clin_lowinf`, `subclin_inf` and
`clin_inf`. They are two axes, clinical status and infectiousness, spelled into state names, and
every rule that depends on one axis has to list the names that share it
(`[c for c in ACTIVE_COMPS if c.startswith("subclin_")]`).

`kiribati_tb/model_strata.py`, the third builder, makes each axis a property. There is one
disease state, `tb`, and only it is stratified by

- `CLIN = Property("clin", ("subclin", "clin"))`, and
- `INF = Property("inf", ("lowinf", "inf"))`.

```python
pmap = (PropertyMap.from_property(STATE)          # mtb_naive … tb … recovered (7 states)
        .stratify(CLIN, where=STATE["tb"])
        .stratify(INF, where=STATE["tb"])
        .stratify(AGE)
        .stratify(REACH))
```

The compartments are the same 160: `state=tb, clin=subclin, inf=lowinf` is the original's
`subclin_lowinf`. So are the equations, and the builder passes the same golden tests as the
other two.

## Each axis moves on its own

::::{grid} 1 1 2 2
:::{grid-item-card} Paired states (`model_summer4.py`)
```python
CLINICAL = (("subclin_inf", "clin_inf"),
            ("subclin_lowinf", "clin_lowinf"))
chain("clinical_progression", CLINICAL,
      Param("clinical_progression_rate"))
```
:::
:::{grid-item-card} An axis (`model_strata.py`)
```python
TransitionFlow(
    "clinical_progression",
    CLIN["subclin"], CLIN["clin"],
    Param("clinical_progression_rate"))
```
:::
::::

A flow whose source and destination name only `CLIN` changes only clinical status: the state,
the infectiousness level, age and reachability are carried along. No list of pairs is needed,
and a third infectiousness level would need no change here at all.

Rules that depend on one axis say so:

| Rule | Four named states | Two properties |
| --- | --- | --- |
| reduced detection when subclinical | `where=STATE.isin(["subclin_lowinf", "subclin_inf"])` | `where=CLIN["subclin"]` |
| infectiousness weights | `(STATE.isin(["subclin_lowinf", "clin_lowinf"]), …)` | `(INF["lowinf"], …)`, `(CLIN["subclin"], …)` |
| self-recovery | from `subclin_lowinf` and `subclin_inf` | from `CLIN["subclin"]` |
| TB deaths | from `clin_lowinf` and `clin_inf`, one rate each | from `CLIN["clin"]`, `Multiply(rate, where=INF[inf])` |

## Flows into the disease: say where they land

A flow from `incipient` to `tb` arrives in a state that has `CLIN` and `INF` where its source
has neither: those are *destination-only* properties. The flow has to say where in the
`CLIN × INF` grid it lands. Progression starts subclinical (a selector) and is split by
infectiousness with a **parameter `split`** (summer4 0.3.0a3):

```python
TransitionFlow(
    "progression", STATE["incipient"], STATE["tb"] & CLIN["subclin"], 1.0,
    split={INF: {"lowinf": 1.0 - Param("progression_prop_infectious"),
                 "inf": Param("progression_prop_infectious")}},
    adjust=by_latency_band("progression"),
)
```

Relapse from treatment goes to one cell (`compartment("subclin_lowinf")`), and so does the seed
in the initial population (`compartment("clin_inf")`). **This is the new way to get it wrong.**
A flow into `tb` that names no cell is shared equally among the four disease compartments, with
no error. The goldens catch it; a model without goldens would not.

## The edges

```{code-cell} ipython3
import numpy as np
import pandas as pd

from kiribati_tb import model, model_strata
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors

_, _, tv_params = get_parameters_and_priors()
strata = model_strata.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)


def cell(row):
    return f"tb[{row['clin']}, {row['inf']}]" if "clin" in row else row["state"]


def edge_summary(name):
    e = strata.model.edges(name)
    rows = e.pmap.to_dicts()
    pairs = [(rows[s], rows[d]) for s, d in zip(np.asarray(e.src_idx), np.asarray(e.dest_idx))]
    frame = pd.DataFrame({"from": [cell(a) for a, _ in pairs], "to": [cell(b) for _, b in pairs],
                          "same age and reach": [a["age"] == b["age"] and a["reach"] == b["reach"]
                                                 for a, b in pairs]})
    return frame.groupby(["from", "to", "same age and reach"]).size().rename("edges")


pd.concat({name: edge_summary(name) for name in [
    "progression", "clinical_progression", "clinical_regression", "infectiousness_gain",
    "infectiousness_loss", "self_recovery", "tx_relapse"]}, names=["flow"]).to_frame()
```

Each move connects exactly the intended cells, 16 times (once per age × reachability stratum),
and never changes age or reachability. `tests/test_model_strata.py` asserts it, including that
TB deaths leave only from `CLIN["clin"]`.

## Same model, different order

The compartments are the same but come in a different order (the map lists `tb`'s four cells
together). Reordering by summer2 name, the two vector fields agree to rounding:

```{code-cell} ipython3
from kiribati_tb.runner_tools import compartment_labels, midpoint_params, run_compartments
from kiribati_tb.variants import PARITY

familiar = model.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
params = midpoint_params()
order = [compartment_labels(strata).index(label) for label in compartment_labels(familiar)]
y = run_compartments(familiar, params, PARITY).loc[2025.0].to_numpy()  # familiar order
y_strata = np.empty_like(y)
y_strata[order] = y
da = np.asarray(familiar.model.vector_field(2025.0, y, params))
db = np.asarray(strata.model.vector_field(2025.0, y_strata, params))[order]
print("max |dy/dt difference| / max |dy/dt| at a 2025 state:", np.max(np.abs(da - db)) / np.max(np.abs(da)))
```

## Outputs keep the original names

Output names built from a state name (`prev_subclin_infXage_15Xreach_reachable`,
`notifications_clin_inf`, the TST and PEARL positivity outputs) cannot come from `by=` here: it
would name them `prev_tbXclin_subclinXinf_inf`. Instead, `request_model_outputs` takes
`compartment=model_strata.compartment`, which maps each original state name to its selector
(`"subclin_inf"` → `STATE["tb"] & CLIN["subclin"] & INF["inf"]`), and `flow_sources`, as in
the compact model. All ~680 names and values are the original's.

New outputs by axis are then one request each:

```python
outputs.compartments("tb_prevalence", STATE["tb"], by=[CLIN, INF, (CLIN, INF)])
outputs["perc_subclinical"] = 100 * outputs.ref("tb_prevalenceXclin_subclin") / outputs.ref("tb_prevalence")
```

## Counts and trade-offs

```{code-cell} ipython3
from pathlib import Path
import ast

from kiribati_tb import model_summer4


def code_lines(path):
    src = Path(path).read_text()
    tree, doc = ast.parse(src), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            doc.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    lines = src.splitlines()
    return len(lines), sum(1 for i, l in enumerate(lines, 1)
                           if l.strip() and not l.strip().startswith("#") and i not in doc)


builders = {"model.py": model, "model_summer4.py": model_summer4, "model_strata.py": model_strata}
pd.DataFrame({
    "states": [10, 10, 7],
    "properties": ["state, age, reach", "state, age, reach", "state, clin, inf, age, reach"],
    "lines": [code_lines(f"../src/kiribati_tb/{n}")[0] for n in builders],
    "code lines": [code_lines(f"../src/kiribati_tb/{n}")[1] for n in builders],
    "flows": [len(m.get_tb_model(DEFAULT_MODEL_CONFIG, tv_params).model.flows)
              for m in builders.values()],
}, index=list(builders))
```

| | For | Against |
| --- | --- | --- |
| **Two properties** | each axis is one flow and one selector; `split=` for the infectious share; outputs by axis for free; a new level or axis is one trait | names differ from the original (labels and state-named outputs need a mapping); every flow into `tb` must name its cell or is split equally without warning; a ragged map (`CLIN["clin"]` matches nothing outside `tb`) |
| **Four named states** | reads like the original; every compartment has a familiar name | each axis is spread over pairs of names; rules list names by string pattern |

Neither is faster: the compartments and the equations are the same. The choice is about which
model is easier to read and to change. For this model, where clinical status and infectiousness
are genuinely independent axes, the properties say what the model means.
