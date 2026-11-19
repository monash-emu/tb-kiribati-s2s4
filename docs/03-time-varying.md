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

# Step 3: time-varying functions

summer2 has three kinds of time-varying input: `Parameter` and `Time` objects combined with
arithmetic, `Function(f, [args])` wrapping a Python function, and interpolators from
`summer2.functions.time`. summer4 has one kind: the **rate expression**. Arithmetic on `Param`
and `Time()` builds one; `Data(...).interp(...)` builds one from data; `defer(f)(args)` wraps
a function whose arguments are expressions.

| summer2 | summer4 |
| --- | --- |
| `Parameter("x")`, `Time` | `Param("x")`, `Time()` |
| `Parameter("a") * Function(f, [Time, Parameter("b")])` | `Param("a") * f(Time(), Param("b"))` when `f` is arithmetic; else `defer(f)(Time(), Param("b"))` |
| `stf.get_linear_interpolation_function(x, y)` | `Data(x, y).interp("linear")` |
| `stf.get_sigmoidal_interpolation_function(x, y)` | `Data(x, y).interp("sigmoidal", sharpness=16)` |
| one interpolator per age, in a dict | `Data.table(times, frame, over=AGE).interp(...)`: one expression, one value per age |

```{code-cell} ipython3
import numpy as np
import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"

from summer4 import OutputSet, Param, SavePlan, Time
from summer4.data import Data

from kiribati_tb.demographic_tools import load_demography
from kiribati_tb.model import AGE, get_neg_tx_outcome_funcs, get_tb_model, tanh_based_scaleup
from kiribati_tb.paths import GOLDEN_FOLDER
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors, midpoint_params

_, _, tv_params = get_parameters_and_priors()
demog = load_demography()
golden = pd.read_parquet(GOLDEN_FOLDER / "demography.parquet").set_index("time")
```

## Interpolation: pass summer2's curvature

summer2's sigmoidal interpolation defaults to curvature 16. summer4's `sharpness` is the same
parameter, but defaults to 1, which is close to linear. Porting the call without the argument
moves the births entry rate between knots, and the population by up to 3.6 people per
compartment after 1950: enough to fail parity. The figure saves the births entry rate both ways
during a solve and draws the original's values over them.

```{code-cell} ipython3
entry = {
    "sharpness=16 (summer2's default)": Data(demog.entry_times, demog.entry_values).interp(
        "sigmoidal", sharpness=16.0
    ),
    "sharpness=1 (summer4's default)": Data(demog.entry_times, demog.entry_values).interp(
        "sigmoidal"
    ),
}
outputs = OutputSet()
for name, expression in entry.items():
    outputs[name] = expression           # any expression of time can be saved

tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
times = golden.index.to_numpy()
params = midpoint_params()
result = tb.model.run(params, t0=1850.0, t1=2035.0, dt=0.5,
                      save=outputs.plan(SavePlan(ts=times)), solver="euler")
named = outputs.evaluate(result, params)
frame = pd.DataFrame({n: np.asarray(named[n].array()) for n in named.keys()}, index=times)
frame["original (summer2)"] = golden["entry_rate"]
frame.loc[1990:2010].plot(title="Births entry rate between the 1990–2010 knots",
                          labels={"index": "year", "value": "people per year", "variable": ""})
```

The sharpness-16 line lies on the original; the sharpness-1 line cuts the corners.

## Functions of expressions: write them with expressions

The original's `tanh_based_scaleup(t, shape, inflection_time, start, end)` is arithmetic and a
`tanh`. Calling it with `Time()` and `Param`s builds the expression directly; no `Function`
wrapper is needed. Only one change: `jnp.tanh` becomes summer4's `tanh`, which accepts
expressions.

## Several values computed together: `field_refs(defer(f)(...), Schema)`

The original's negative treatment outcomes are one Python function per age that computes the
natural-death share, the TB-death share and the relapse share together, and returns a dict of
two rates. summer4 runs the same function once for every age: `defer` wraps it, its arguments are
expressions (the per-age death table, the treatment success interpolation, two parameters), and
`field_refs` names the fields of what it returns.

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
def get_neg_tx_outcome_funcs(bckd_death_funcs, tsr):
    def get_neg_tx_outcome_funcs_age(mu, tsr, dur, pct):
        ...
        return {"relapse": ..., "death": ...}
    return {
        age: Function(get_neg_tx_outcome_funcs_age,
            [mu, tsr, Parameter("tx_duration"),
             Parameter("pct_neg_tx_death")])
        for age, mu in bckd_death_funcs.items()}
```
:::
:::{grid-item-card} summer4
```python
class NegTxOutcomes(NamedTuple):
    relapse: Any
    death: Any

def get_neg_tx_outcomes(mu, tsr, dur, pct):
    ...   # the same body, for every age at once
    return NegTxOutcomes(relapse=..., death=...)

def get_neg_tx_outcome_funcs(bckd_death_funcs, tsr):
    computed = defer(get_neg_tx_outcomes)(
        bckd_death_funcs, tsr,
        Param("tx_duration"), Param("pct_neg_tx_death"))
    return field_refs(computed, NegTxOutcomes)
```
:::
::::

`outcomes.relapse` and `outcomes.death` are ordinary rate expressions with one value per age.
The function runs once per vector-field evaluation however many flows read its fields. Inside
it, the per-age death rate arrives as a summer4 `GroupedRate`: use NumPy functions
(`np.exp`, `np.maximum`) and arithmetic on it, not `jnp` functions, which do not know the type.
(The original's `jnp.max(jnp.array((a, 0.)))` also had to become `np.maximum(a, 0.)` to work
on all ages at once.)

The next cell saves the port's death rates and treatment outcomes during a solve and compares
them with the original's, evaluated by summer2 on the same half-year grid:

```{code-cell} ipython3
tsr = tv_params["tx_success_pct"]
success = Data(tsr.index.to_numpy(float), tsr.to_numpy(float) / 100.0).interp("linear")
death = demog.death_rate_table(AGE).interp("sigmoidal", sharpness=16.0)
outcomes = get_neg_tx_outcome_funcs(death, success)
detection = Param("recent_detection_rate") * tanh_based_scaleup(
    Time(), Param("passive_detection_shape"), Param("passive_detection_inflection"),
    Param("passive_detection_past_frac"))

probe = OutputSet()
probe["death_rate"] = death               # one series per age (over=AGE is inferred)
probe["tx_relapse"] = outcomes.relapse
probe["tx_death"] = outcomes.death
probe["tx_success"] = success
probe["passive_detection_rate"] = detection
gold_params = __import__("yaml").safe_load(
    (GOLDEN_FOLDER / "hetero_baseline" / "params.yaml").read_text())["params"]
result = tb.model.run(gold_params, t0=1850.0, t1=2035.0, dt=0.5,
                      save=probe.plan(SavePlan(ts=times)), solver="euler")
saved = probe.evaluate(result, gold_params)

rows = {}
for name in ("death_rate", "tx_relapse", "tx_death"):
    got = np.asarray(saved[name].array())
    want = golden[[f"{name}_{age}" for age in AGE.traits]].to_numpy()
    rows[name] = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12))
for name in ("tx_success", "passive_detection_rate"):
    got = np.asarray(saved[name].array())
    rows[name] = np.max(np.abs(got - golden[name].to_numpy()) / np.abs(golden[name].to_numpy()))
pd.Series(rows, name="largest relative difference from summer2")
```

All five agree with the original to rounding error. `tests/test_inputs.py` asserts it.
