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

# Step 1: model structure and stratification

summer2 builds a model from a list of compartment names, then splits it with stratification
objects. summer4 has the same two steps with different nouns: a **property** with its
**traits** for each way of classifying people, and a **property map** that holds the
compartments as every combination of traits.

## Compartments become a property

::::{grid} 1 1 2 2
:::{grid-item-card} summer2 (`tbh/model.py`)
```python
COMPARTMENTS = ("mtb_naive", "incipient", ...,
                "treatment", "recovered")

model = CompartmentalModel(
    times=(1850, 2035),
    compartments=COMPARTMENTS,
    infectious_compartments=ACTIVE_COMPS,
)
```
:::
:::{grid-item-card} summer4 (`kiribati_tb/model.py`)
```python
COMPARTMENTS = ("mtb_naive", "incipient", ...,
                "treatment", "recovered")
STATE = Property("state", COMPARTMENTS)
AGE = Property("age", AGE_GROUPS)
REACH = Property("reach", ("reachable", "unreachable"))

model = FlowModel(PropertyMap.from_property(STATE))
```
:::
::::

Three things move. The time window is not part of the model: it is given to `run`, so one
compiled model can run any window. The infectious compartments are not part of the model either:
they belong to the force of infection that uses them ([step 4](04-mixing-and-infection)). And
the strata are properties declared up front, so flows, outputs and adjustments can refer to
`AGE["15"]` before the model is stratified.

The property names are worth choosing with care. summer4 names a stratum output
`{name}X{property}_{trait}` ([step 5](05-derived-outputs)); calling the properties `age` and
`reach` makes those names the original's (`populationXage_15Xreach_reachable`) for free.
(summer2's stratification was called `reachability`; its output names used `reach`.)

```{code-cell} ipython3
from kiribati_tb.model import (
    AGE, REACH, STATE, get_natural_tb_model, add_detection_and_treatment,
    stratify_model_by_age, stratify_model_by_reachability, get_neg_tx_outcome_funcs,
)
from kiribati_tb.demographic_tools import load_demography
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors

_, _, tv_params = get_parameters_and_priors()
demog = load_demography()
model, infection_flows = get_natural_tb_model(DEFAULT_MODEL_CONFIG, demog.total)
print(model.pmap.size, "compartments:", model.pmap.labels())
```

## Stratification is a method on the model

The port keeps the original's order: build the natural history, add detection and treatment,
stratify by age, stratify by reachability, add births and deaths. summer4 allows this because
flows are re-resolved against the final map when the model is compiled.

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
age_strat = AgeStratification(
    name="age", strata=age_groups,
    compartments=COMPARTMENTS)
age_strat.set_flow_adjustments(
    "infection_from_mtb_naive",
    {age: Parameter("rel_sus_children")
          if int(age) < 15 else 1.
     for age in age_groups})
model.stratify_with(age_strat)
```
:::
:::{grid-item-card} summer4
```python
model.stratify(AGE)
model.add_flow(TransitionFlow(
    "ageing", AGE.present(), AGE.present(), 1.0,
    pairing=TraitChain.from_breakpoints(AGE)))
model.adjust_flow(
    "infection_from_mtb_naive",
    Multiply(Param("rel_sus_children"),
             where=CHILDREN))
```
:::
::::

- **Ageing is a flow you declare.** summer2's `AgeStratification` adds ageing between
  consecutive bands at one over the band width. In summer4 that is one `TransitionFlow` whose
  `pairing=TraitChain.from_breakpoints(AGE)` moves each band to the next at the rate the
  breakpoints imply. A plain `model.stratify(AGE)` would add no ageing.
- **An adjustment names where it applies.** `set_flow_adjustments` takes one value per
  stratum, with `1.` for "unchanged". `Multiply(value, where=selector)` applies only where the
  selector matches and leaves the rest alone, so "children's susceptibility" is one line.
  Several adjustments to one flow are separate `Multiply` items.

```{code-cell} ipython3
from summer4.data import Data

tsr = tv_params["tx_success_pct"]
time_variant_tsr = Data(tsr.index.to_numpy(float), tsr.to_numpy(float) / 100).interp("linear")
death = demog.death_rate_table(AGE).interp("sigmoidal", sharpness=16.0)
screening_flows, _ = add_detection_and_treatment(model, time_variant_tsr, [])
stratify_model_by_age(model, get_neg_tx_outcome_funcs(death, time_variant_tsr), [])
print("after age:", model.pmap.size)
stratify_model_by_reachability(model, infection_flows, screening_flows)
print("after reachability:", model.pmap.size)
model.pmap.labels()[:4]
```

## The starting population

summer2 sets the unstratified distribution, then each stratification splits it: age evenly by
default, reachability by `set_population_split`. summer4 declares the base values and the
splits in one `InitialPopulation`; a property with no `Split` is shared out evenly, as summer2's
age default does.

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
model.set_initial_population(distribution={
    "mtb_naive": pop.iloc[0] - seed,
    "clin_inf": seed})
...
reach_strat.set_population_split({
    "reachable": Parameter("reachable_pop_frac"),
    "unreachable": 1. - Parameter("reachable_pop_frac")})
```
:::
:::{grid-item-card} summer4
```python
model.set_initial_population(InitialPopulation(
    {STATE["mtb_naive"]: pop.iloc[0] - seed,
     STATE["clin_inf"]: seed},
    splits=(Split(REACH, {
        "reachable": Param("reachable_pop_frac"),
        "unreachable": REMAINDER}),),
))
```
:::
::::

`REMAINDER` is "whatever is left", so the unreachable share follows the parameter without a
second expression. The parity tests check that the 1850 population equals the original's to the
last bit.

## Compiling

`model.compile()` turns the declarations into a `CompiledModel` that runs under `jit`. It is
also when every selector is checked against the final map, so a typo in a trait name fails
here, not silently. `get_tb_model` returns the compiled model and its outputs:

```{code-cell} ipython3
from kiribati_tb.model import get_tb_model

tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
print(type(tb.model).__name__, tb.model.pmap.size, "compartments,",
      len(tb.model.flows), "flows,", len(tb.outputs.kept()), "kept outputs")
```
