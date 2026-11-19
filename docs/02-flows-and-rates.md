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

# Step 2: flows and rates

## Transition flows

Every `add_transition_flow` becomes a `TransitionFlow` with the same name. The source and
destination are selectors on the state property; the rate is a rate expression.

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
model.add_transition_flow(
    name="clearance",
    fractional_rate=Parameter("clearance_rate"),
    source="contained",
    dest="cleared",
)
```
:::
:::{grid-item-card} summer4
```python
model.add_flow(TransitionFlow(
    "clearance",
    STATE["contained"],
    STATE["cleared"],
    Param("clearance_rate"),
))
```
:::
::::

Keep the original's flow names, typos included (`infectiousnnes_gain_clin`): outputs, tests and
anyone reading both codebases refer to flows by name.

## `source_strata` and `dest_strata` become selectors

summer2 restricts a flow with `source_strata={"age": "15"}`. A summer4 selector says the same
thing and composes with `&`, `|` and `~`. Deaths in the original go from every compartment
except (`mtb_naive`, age 0) to (`mtb_naive`, age 0) in the same reachability stratum:

::::{grid} 1 1 2 2
:::{grid-item-card} summer2: 79 flows
```python
for compartment in COMPARTMENTS:
    for i_age, source_age in enumerate(age_groups):
        if i_age > 0 or compartment != "mtb_naive":
            model.add_transition_flow(
                name=f"all_cause_mortality_from_"
                     f"{compartment}_age_{source_age}",
                fractional_rate=bckd_death_funcs[source_age],
                source=compartment,
                source_strata={"age": source_age},
                dest="mtb_naive",
                dest_strata={"age": age_groups[0]})
```
:::
:::{grid-item-card} summer4: one flow
```python
NEWBORN = STATE["mtb_naive"] & AGE["0"]

model.add_flow(TransitionFlow(
    "all_cause_mortality",
    ~NEWBORN,       # every other compartment
    NEWBORN,        # same reachability: unmatched
    bckd_death_funcs,  # per-age table over AGE
))
```
:::
::::

The death rate is a table with one column per age band (`Data.table(..., over=AGE)`), so one
flow carries every age's rate. Properties the destination selector does not mention
(reachability) are kept, which is summer2's "dest strata default to the source's". The
original's 79 per-(compartment, age) names survive as outputs: each selects that flow's source
([step 5](05-derived-outputs)). The same goes for TB deaths (one flow per form of disease, not
per age) and deaths on treatment.

## Entry flows

`add_importation_flow(..., dest_strata={...})` becomes an `EntryFlow` into a selector:

```python
frac = Param("reachable_pop_frac")
for reach, share in (("reachable", frac), ("unreachable", 1.0 - frac)):
    model.add_flow(EntryFlow(f"births_{reach}", NEWBORN & REACH[reach], entry_rate * share))
```

## Placeholder rates and age-specific rates

The original declares `tx_relapse` with `fractional_rate=1.` and replaces it per age in the age
stratification. The port does the same with `adjust_flow`, multiplying by a per-age value:

```python
model.add_flow(TransitionFlow("tx_relapse", STATE["treatment"], STATE["subclin_lowinf"], 1.0))
...
model.adjust_flow("tx_relapse", Multiply(neg_tx_outcome_funcs.relapse))   # per age
```

`neg_tx_outcome_funcs.relapse` is a rate expression with one value per age band, so no `where=`
is needed. Where a rate is known when the flow is declared, give it directly; the placeholder
pattern is only worth keeping when it mirrors the original.

## Rates are expressions

A summer4 rate is anything built from numbers, `Param`, `Time()`, data interpolations and the
arithmetic between them. The detection flows show the pattern: the original multiplies a
`Function` by a parameter; the port multiplies expressions.

```{code-cell} ipython3
from summer4 import Param, Time
from kiribati_tb.model import tanh_based_scaleup

tv_detection_rate = Param("recent_detection_rate") * tanh_based_scaleup(
    Time(),
    Param("passive_detection_shape"),
    Param("passive_detection_inflection"),
    Param("passive_detection_past_frac"),
    1.0,
)
tv_detection_rate
```

`tanh_based_scaleup` is the original's function, unchanged except that it calls summer4's
`tanh`: given expressions, it returns an expression. Rate expressions also hoist: anything that
depends only on parameters is computed once per run, before the solve.

## Checking the flows

The compiled model lists every flow. Comparing names with the original's is a quick first check
(the full check is the golden comparison in [step 7](07-validation)).

```{code-cell} ipython3
from kiribati_tb.model import get_tb_model
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors

_, _, tv_params = get_parameters_and_priors()
tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
sorted(tb.model.flows)
```
