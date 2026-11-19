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

# Step 7: validate against the goldens

The port is correct when it reproduces the original's numbers. `tests/test_parity.py` checks,
for each of the five fixtures:

1. the 160 compartment names, and the 1850 population to the last bit;
2. every compartment at every year, to `1e-5` relative (or `1e-6` people absolute);
3. the set of output names: exactly the original's, including every `save_results=False`
   intermediate (`by=` groupings aside);
4. every output at every year, to `1e-5` relative (or `1e-8` absolute).

The comparison helpers in `tests/parity.py` report the largest absolute and relative error per
column rather than pass/fail, so a failure says where. This page runs the same comparison and
shows the worst columns.

```{code-cell} ipython3
import pandas as pd

from kiribati_tb.model import get_tb_model
from kiribati_tb.runner_tools import (
    DEFAULT_MODEL_CONFIG, get_parameters_and_priors, run_compartments, run_model,
)
from kiribati_tb.scenarios import SCENARIOS_BY_ID
from kiribati_tb.variants import PARITY
import sys
sys.path.insert(0, "../tests")
from parity import parity_report, read_golden, read_params

FIXTURES = {
    "no_transmission": (False, None),
    "homog_baseline": (False, None),
    "hetero_baseline": (True, None),
    "hetero_scenario_3": (True, "scenario_3"),
    "hetero_scenario_8": (True, "scenario_8"),
}
_, _, tv_params = get_parameters_and_priors()
rows, worst = [], {}
for fixture, (hetero, scenario) in FIXTURES.items():
    programs = SCENARIOS_BY_ID[scenario].scr_prgs if scenario else []
    tb = get_tb_model(DEFAULT_MODEL_CONFIG | {"heterogeneous_mixing": hetero}, tv_params,
                      programs, PARITY)
    params = read_params(fixture)
    comps = parity_report(run_compartments(tb, params, PARITY),
                          read_golden(fixture, "compartments"), atol=1e-6)
    golden = read_golden(fixture, "derived")
    if fixture == "no_transmission":   # ratios of solver noise once the seed has died out
        golden = golden.drop(columns=[c for c in golden if c.startswith(("perc_prev_", "perc_notif"))])
    derived = run_model(tb, params, PARITY, names=list(tb.outputs.keys()))
    outs = parity_report(derived, golden, atol=1e-6 if fixture == "no_transmission" else 1e-8)
    worst[fixture] = outs.head(3)
    rows.append({
        "fixture": fixture,
        "compartments": len(comps),
        "worst compartment rel. error": comps["max_rel"].max(),
        "worst compartment abs. error": comps["max_abs"].max(),
        "outputs compared": len(outs),
        "worst output rel. error": outs["max_rel"].max(),
        "all pass": bool(comps["passes"].all() and outs["passes"].all()),
    })
pd.DataFrame(rows).set_index("fixture")
```

Every compartment and every output agrees with summer2 to about `1e-7` relative, a hundred times
inside the bar. The exception is in appearance only: without transmission the disease states
hold about `1e-10` people (and go negative) once the seed has died out, in both models, so their
relative differences are ratios of solver noise. They pass on the absolute tolerance. The
largest absolute difference of any compartment, in any fixture, is under a thousandth of a
person. What remains is the two solvers' own error at `1e-10`: summer2gen's `odeint` and
diffrax's `Dopri5` take different steps.

The worst outputs for the baseline are prevalences of the earliest infection state, the
smallest and fastest-changing compartments:

```{code-cell} ipython3
worst["hetero_baseline"]
```

## Checking pieces before the whole

The whole-model comparison says *whether* the port is right. When it is not, the piece-by-piece
goldens say *where*. `tests/test_inputs.py` compares, on their own:

- the per-age death rates, births entry rate, treatment success, treatment outcomes and passive
  detection rate, saved by expression during a solve ([step 3](03-time-varying));
- the yearly mixing matrices for five parameter draws at ten times, the spectral radius, the
  matrix's gradient and its distance to `conmat` ([step 4](04-mixing-and-infection)).

The `no_transmission` fixture then checks structure, ageing, the care cascade, deaths and births
with infection switched off, before transmission and mixing are compared.

## Running the tests

```bash
devq build -- pixi run test       # about 40 s
```
