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

# Step 6: computed values

summer2 has *computed values*: functions of time, parameters and state that the model evaluates
at every step, which flows can use and outputs can save. The Kiribati model saves three:

- `passive_detection_rate_clin` and `passive_detection_rate_subclin`, the time-varying passive
  detection rates its detection flows use;
- `mixing_matrix_distance`, the Canberra distance from the year's mixing matrix to the `conmat`
  synthetic matrix, which the calibration scores.

## In summer4 a computed value is a rate expression, saved by assignment

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
model.add_computed_value_func(
    "passive_detection_rate_clin", tv_detection_rate)
model.add_computed_value_func(
    "passive_detection_rate_subclin",
    Parameter("rel_detection_subclin") * tv_detection_rate)
matrix_dist = Function(canberra_distance,
                       [age_mixing_matrix, conmat_matrix])
model.add_computed_value_func(
    "mixing_matrix_distance", matrix_dist)
...
for comp_val in computed_values_to_save:
    model.request_computed_value_output(comp_val)
```
:::
:::{grid-item-card} summer4
```python
computed_values = {
    "passive_detection_rate_clin": tv_detection_rate,
    "passive_detection_rate_subclin":
        Param("rel_detection_subclin") * tv_detection_rate,
    "mixing_matrix_distance": defer(canberra_distance)(
        age_mixing_matrix, conmat_matrix),
}
...
for name, expression in computed_values.items():
    outputs[name] = expression
```
:::
::::

`outputs[name] = expression` saves the value of any expression that reads time, state or flows,
during the solve, at the save times. `tv_detection_rate` is the same object the detection flows
use, and `age_mixing_matrix` the same object the force of infection uses, so saving them costs
one evaluation per save time and duplicates no code.

An expression that reads only parameters is a constant, not a time series, and assigning one
raises an error that says so. (Multiply a saved output by it instead.)

## What this replaces

Before summer4's step 40, the first Kiribati port could not save an ungrouped rate expression.
It wrote the three values as `SaveFn` leaves that re-implemented the detection rate's tanh
formula and the mixing lookup in plain JAX, from the save time and the prepared parameters, and
tested them against the goldens to make sure the copies matched (its workaround W2). `SaveFn`,
`ComputedValue` and `compile(derived_fn=...)` no longer exist.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import plotly.io as pio
import yaml

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"

from kiribati_tb.model import get_tb_model
from kiribati_tb.paths import GOLDEN_FOLDER
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors, run_model
from kiribati_tb.variants import PARITY

_, _, tv_params = get_parameters_and_priors()
tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
fixture = GOLDEN_FOLDER / "hetero_baseline"
params = yaml.safe_load((fixture / "params.yaml").read_text())["params"]
golden = pd.read_parquet(fixture / "derived.parquet").set_index("time")

names = ["passive_detection_rate_clin", "passive_detection_rate_subclin", "mixing_matrix_distance"]
got = run_model(tb, params, PARITY, names=names)
pd.Series({n: float(np.max(np.abs(got[n] - golden[n]) / np.abs(golden[n]))) for n in names},
          name="largest relative difference from summer2")
```

```{code-cell} ipython3
frame = got[["mixing_matrix_distance"]].rename(columns={"mixing_matrix_distance": "summer4"})
frame["summer2 (golden)"] = golden["mixing_matrix_distance"]
frame.plot(title="Distance from the yearly mixing matrix to conmat",
           labels={"time": "year", "value": "Canberra distance", "variable": ""})
```

The distance is flat before 1950, where the UN population data start and the matrix is held at
its first year, then moves with the population's age structure.

## Values that several rates share

When a function computes several values that rates need, summer2 used a `Function` per value, or
summer4's former `derived_fn`. summer4 now reads them through
`field_refs(defer(f)(...), Schema)`, as the treatment outcomes do in
[step 3](03-time-varying). With the arguments `Params(), y=StateArray(), t=Time()` the function
receives the whole parameter set, the state vector and the time, which is exactly what
`derived_fn` received:

```python
D = field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)
model.add_flow(TransitionFlow("x", STATE["a"], STATE["b"], D.some_rate))
outputs["some_rate"] = D.some_rate        # and save it
```

`f` runs once per vector-field evaluation however many rates read `D`, and parameter-only parts
of the model are still hoisted. The Kiribati model needs only the narrower form: its treatment
outcomes read a table, an interpolation and two parameters, not the whole parameter set.
