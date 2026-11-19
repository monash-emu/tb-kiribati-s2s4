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

# Step 8: parity, then improve

Parity is the definition of a correct port, but some of what parity reproduces is a choice of
summer2's that summer4 does not have to make. `kiribati_tb.variants` defines two configurations
of the **same model**: `PARITY`, which matches the goldens, and `IMPROVED`, which changes how
the equations are solved and what the flow outputs report.

| Choice | `PARITY` | `IMPROVED` |
| --- | --- | --- |
| Flow output values | `"midpoint"`: the average of the flow's rate at this save time and the previous one | `"rate"`: the instantaneous rate at the save time, people per year |
| Cumulative outputs | `cumsum` of those values from 2026 | `cumulative_integral` of the rate from 2026 |
| Save grid | yearly | every 0.1 year, plus the screening ramp times |
| Solver | Dopri5, `rtol = atol = 1e-10` | Tsit5, PID controller at `rtol = 1e-6, atol = 1e-9`, told where the vector field jumps (`jump_ts`) |
| Computed values | saved expressions | saved expressions (both variants use summer4 idiom) |

```python
tb = get_tb_model(config, tv_params, programs, IMPROVED)   # rate-based flow outputs
derived = run_model(tb, params, IMPROVED)                  # its solver and save grid
```

The comparisons below use the priors' midpoints and are produced by
`kiribati_tb.comparison` (also `pixi run compare`, which writes the tables to
`outputs/compare/`).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"
pd.options.display.float_format = "{:.4g}".format

from kiribati_tb.comparison import (
    annual_flow_comparison, build, cumulative_comparison, solver_comparison,
)
from kiribati_tb.runner_tools import midpoint_params
from kiribati_tb.variants import IMPROVED, PARITY

params = midpoint_params()
parity_tb, improved_tb = build(PARITY), build(IMPROVED)
```

## 1. The solver: as accurate as needed, for a sixth of the steps

The vector field jumps at every calendar year, when the mixing matrix changes. An adaptive
solver that does not know this keeps proposing steps across a jump, failing its error test, and
retrying smaller. summer2's parity tolerance (`1e-10`) hides the jumps by brute force. The
improved solver gives the jump times to the step-size controller as `jump_ts`, so each step ends
exactly on a jump. Each solver is measured against a reference solve at `1e-12`:

```{code-cell} ipython3
solvers = solver_comparison(parity_tb, params)
solvers
```

- The parity setting is the most accurate and the most expensive: six times the steps of the
  improved solver, and 40% of them rejected.
- summer2's default tolerance (`1.4e-4`, what the original calibrated with) is cheapest, but its
  compartments are off by up to 0.4%.
- summer4's default (Tsit5, `rtol = 1e-6`) is 10 times more accurate than that, and telling it
  about the jumps cuts its rejected steps by two thirds and its error by another factor of 20.

The improved solver's 2e-5 worst-case error is far below anything the calibration data can
resolve, so it is the better default for calibration and scenario runs; parity is for checking
the port.

## 2. What a yearly flow output means

A yearly output such as "TB incidence in 2030" should be the number of people who developed TB
during the year. The model's flows are instantaneous rates, so that number is the **integral of
the rate over the year**. summer2's default flow output is the midpoint convention: the average
of the rate at the start and end of the year, a trapezoid with one panel. That is a good
approximation when the rate changes slowly and a poor one when it does not.

`annual_flow_comparison` computes the accurate yearly number (the rate saved every 0.02 years
and at the screening ramp times, integrated) and sets beside it:

- **parity**: the midpoint value;
- **improved_rate**: the improved variant's output at the save time, the instantaneous rate
  (people per year at that moment, which is a different quantity, not a yearly count);
- **improved_interval**: the yearly integral from the improved variant's own 0.1-year grid
  (the yearly differences of its `cumulative_integral`).

### Baseline

```{code-cell} ipython3
baseline = annual_flow_comparison(parity_tb, improved_tb, params)
errors = ["parity_err_%", "improved_rate_err_%", "improved_interval_err_%"]
baseline.groupby("output")[errors].agg(lambda s: s.abs().max()).drop(index="screening")
```

Over 2020–2035, without screening, the midpoint value is within 0.35% of the accurate yearly
count: the rates change slowly. The improved interval is a hundred times closer (0.003%). The
instantaneous rate differs by up to 11%: it is a rate at a moment, not a count, and should be
read as such.

### With screening

Screening campaigns switch on and off within a year. Scenario 3 screens 84.5% of the population
during 2026, ramping up in the last 0.01 year of 2025 and down in the last 0.01 year of 2026.

```{code-cell} ipython3
screening_tb = {v.name: build(v, "scenario_3") for v in (PARITY, IMPROVED)}
scenario = annual_flow_comparison(screening_tb["parity"], screening_tb["improved"], params)
scenario.loc[(["screening", "notifications", "tb_incidence"], [2026.0, 2027.0, 2028.0]),
             ["accurate", "parity", "improved_interval", "parity_err_%", "improved_interval_err_%"]]
```

The save at 2026.0 falls just after the campaign has switched on, at its full starting rate,
and the save at 2027.0 just after it has switched off. The midpoint averages the rate at those
instants, so it books half the *starting* rate into the year ending 2026 and half into 2027:
about 22,000 people screened and treated in each year, against 221 (0.01 year of screening) and
15,581 (the campaign, during which the screened compartments empty). The campaign total is
overstated 2.8-fold and split across the wrong years. The same misplacement runs through
notifications (41% too high in 2027), incidence (31%) and deaths (19%). The improved variant,
which saves on the ramp times, is within 0.7%.

```{code-cell} ipython3
plot = scenario.reset_index()
plot = plot[plot["output"] == "notifications"].set_index("year")[
    ["accurate", "parity", "improved_interval"]].loc[2022:2032]
plot.plot(title="Notifications per year, scenario 3 (screening in 2026)",
          labels={"value": "people per year", "variable": ""}, markers=True)
```

The remaining 0.3–0.7% in 2027 is the 0.1-year grid during the campaign, when the screened
compartments empty within weeks. A finer grid (`IMPROVED.with_save_step(0.02)`) closes it. summer4
has no exact accumulator of a flow over an interval yet (`futureplans/state-ledgers-flow-integral.md`),
so the save grid sets the accuracy of any count.

## 3. Cumulative outputs from 2026

summer2's `request_cumulative_output(start_time=2026)` sums the yearly flow outputs from the one
labelled 2026, which is the midpoint value for the interval 2025–2026. So "cumulative incidence
from 2026" at 2027 counts two years. `cumulative_integral(start=2026)` is the integral of the
rate from 2026, which is 0 in 2026:

```{code-cell} ipython3
cumulative_comparison(parity_tb, improved_tb, params)
```

The parity column runs about one year's incidence ahead of the improved one throughout. Averted
burden (the difference between two scenarios) is unaffected when both scenarios are identical
before 2026, as here.

## Which is more correct

For the quantities the outputs are named after (people per year, people since 2026):

- **The improved values are closer**: 0.003% from the accurate yearly counts without
  screening, 0.7% with it, against 0.35% and 40% for the midpoint convention, and cumulative
  outputs that start where they say.
- **The improved solver is more accurate per unit of work**: 2e-5 worst case for a sixth of
  the parity solver's steps, against 4e-3 for summer2's default tolerance.

Parity remains the way to show a port is right, and the way to reproduce published numbers.
For new analyses, the improved configuration measures what it claims to.
