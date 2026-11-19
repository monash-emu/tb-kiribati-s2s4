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

# Step 5: derived outputs

summer2 registers derived outputs on the model, one `request_*` call each, and evaluates them
inside `model.run`. summer4 collects them in an **`OutputSet`**, separate from the model, and
evaluates it with `run(..., outputs=outputs)`. The requests are methods with the same shape as
summer2's, so `tbh/outputs.py` ports almost line for line.

## The mapping

| summer2 | summer4 | Stands for |
| --- | --- | --- |
| `request_output_for_compartments(n, comps, strata={"age": a})` | `outputs.compartments(n, STATE.isin(comps) & AGE[a])` | `Compartments().select(where).sum_by()` |
| `request_output_for_flow(n, flow, source_strata={...})` | `outputs.flow(n, flow, source=selector)` | `FlowMass(flow).select(Source(...)).sum_by()` |
| `request_aggregate_output(n, sources)` | `outputs.aggregate(n, sources)` | `ref(a) + ref(b) + …` |
| `request_function_output(n, DerivedOutput("a") / DerivedOutput("b"))` | `outputs[n] = outputs.ref("a") / outputs.ref("b")` | the expression |
| `request_function_output(n, Function(f, [...]))` | `outputs.function(n, f, "a", Param("x"))` | `defer(f)(ref("a"), Param("x"))` |
| `request_cumulative_output(n, src, start_time=2026)` | `outputs.cumsum(n, src, start=2026)` (amounts) or `outputs.cumulative_integral(...)` (rates) | `ref(src).cumsum(start=)` |
| `request_computed_value_output(n)` | `outputs[n] = <rate expression>` | a saved expression ([step 6](06-computed-values)) |
| `save_results=False` | `keep=False` | computed, not returned |
| a loop over strata building `…Xage_{a}Xreach_{r}` | `by=[AGE, REACH, (AGE, REACH)]` | one grouped sum per grouping |
| `model.run(params)`; `model.get_derived_outputs_df()` | `cm.run(params, …, outputs=outputs)` | `outputs.evaluate(cm.run(…, save=outputs.plan()), params)` |
| flow outputs averaged over the step (summer2 default) | `OutputSet(flow_values="midpoint")` | `.midpoint()` on each flow output |

## Side by side

::::{grid} 1 1 2 2
:::{grid-item-card} summer2: prevalence by age and reachability (20 lines)
```python
for comp in compartments:
    for age in age_strata:
        model.request_output_for_compartments(
            name=f"prev_{comp}Xage_{age}",
            compartments=comp, strata={"age": age},
            save_results=False)
        for reach in reach_strata:
            model.request_output_for_compartments(
                name=f"prev_{comp}Xage_{age}Xreach_{reach}",
                compartments=comp,
                strata={"age": age, "reachability": reach},
                save_results=False)
    model.request_aggregate_output(
        name=f"prev_{comp}",
        sources=[f"prev_{comp}Xage_{age}" for age in age_strata],
        save_results=False)
    for reach in reach_strata:
        model.request_aggregate_output(
            name=f"prev_{comp}Xreach_{reach}",
            sources=[f"prev_{comp}Xage_{age}Xreach_{reach}"
                     for age in age_strata],
            save_results=False)
```
:::
:::{grid-item-card} summer4: the same 270 outputs (2 lines)
```python
for comp in compartments:
    outputs.compartments(
        f"prev_{comp}", STATE[comp],
        by=[AGE, REACH, (AGE, REACH)], keep=False)
```
:::
::::

`by=` adds one output per stratum of each grouping, named `{name}X{property}_{trait}`, joined
for a tuple: `prev_incipientXage_15`, `prev_incipientXreach_reachable`,
`prev_incipientXage_15Xreach_reachable`. Those are the original's names because the properties
are called `age` and `reach`. Each grouping is computed once as a small grouped array (a hidden
output such as `prev_incipient[age,reach]`), and each stratum is a row of it.

::::{grid} 1 1 2 2
:::{grid-item-card} summer2: incidence
```python
for inf_cat in ["lowinf", "inf"]:
    model.request_output_for_flow(
        name=f"tb_incidence_{inf_cat}",
        flow_name=f"progression_{inf_cat}")
    for reach in reach_strata:
        model.request_output_for_flow(
            name=f"tb_incidence_{inf_cat}Xreach_{reach}",
            flow_name=f"progression_{inf_cat}",
            source_strata={"reachability": reach},
            save_results=False)
...
request_per_capita_output(model, "tb_incidence",
                          per=100000.)
model.request_cumulative_output(
    name="cum_tb_incidence", source="tb_incidence",
    start_time=2026)
```
:::
:::{grid-item-card} summer4: incidence
```python
for inf_cat in ["lowinf", "inf"]:
    outputs.flow(f"tb_incidence_{inf_cat}",
                 f"progression_{inf_cat}")
    for reach in reach_strata:
        outputs.flow(
            f"tb_incidence_{inf_cat}Xreach_{reach}",
            f"progression_{inf_cat}",
            source=REACH[reach], keep=False)
...
request_per_capita_output(outputs, "tb_incidence",
                          per=100000.0)
cumulative("cum_tb_incidence", "tb_incidence",
           start=CUMULATIVE_START)
```
:::
::::

(`cumulative` is `outputs.cumsum` in the parity variant and `outputs.cumulative_integral` in the
improved one: [step 8](08-parity-and-improved).)

Some loops stay loops, deliberately: a fan-out names strata `X{property}_{trait}`, and the
original's mortality outputs are named `all_cause_mortality_from_{comp}_age_{age}`, which no
naming template produces without renaming every other output. A loop of one-line requests is
fine.

## Intermediates: `keep=False`

`keep=False` is summer2's `save_results=False`: the output is computed if another output reads
it, but is not returned. `outputs.kept()` lists what is returned; `outputs.keys()` lists
everything. `subset(names)` keeps only `names`, which is what a calibration or a posterior run
wants ([step 9](09-calibration)). The first port of this model had to write its own
`restrict_outputs`, inlining intermediates into their readers, for exactly this (its workaround
W5).

```{code-cell} ipython3
from kiribati_tb.model import get_tb_model
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors

_, _, tv_params = get_parameters_and_priors()
tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
outputs = tb.outputs
grouped = [n for n in outputs.keys() if "[" in n]
print(len(outputs.keys()), "names in all:", len(outputs.kept()), "kept,",
      len(outputs.keys()) - len(outputs.kept()) - len(grouped), "intermediates,",
      len(grouped), "hidden groupings")
outputs.kept()[:12]
```

## Running

One call solves the model for exactly what the outputs need (one shared compartment save, and
one save per flow that an output reads) and returns the evaluated outputs:

```{code-cell} ipython3
import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"

from kiribati_tb.runner_tools import midpoint_params, run_model
from kiribati_tb.variants import PARITY

params = midpoint_params()
derived = run_model(tb, params, PARITY)      # tb.model.run(params, ..., outputs=...) inside
derived[["tb_incidence_per100k", "tb_prevalence_per100k", "notifications"]].loc[1950:].plot(
    title="Baseline, priors at their midpoints", labels={"value": "", "variable": ""})
```

`run_model` is a thin wrapper over `CompiledModel.run`: it passes the variant's solver and save
grid, then turns the named result into the original's wide `derived_outputs` frame:

```python
plan = outputs.plan(SavePlan(ts=variant.save_times(tb.breakpoints)))
result = tb.model.run(params, t0=1850.0, t1=2035.0, dt=1.0, save=plan, **variant.run_kwargs())
named = outputs.evaluate(result, params)
frame = named.to_frame(shape="wide", backend="pandas")
```

Pass `names=list(outputs.keys())` to `evaluate` (or `run_model`) to get the intermediates as
well; the parity tests do, to compare every one of the original's names.

## An empty aggregate

The original aggregates the screening flows into `screening`, and with no screening that list
is empty; summer2 returns zeros. summer4's `aggregate` refuses an empty list (it cannot know the
shape), so the port writes the zeros: `outputs["screening"] = 0.0 * outputs.ref("births")`.
