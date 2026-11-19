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

# Step 4: mixing and the force of infection

## The generalised force of infection

The original infects with summer2's `add_infection_generalised_flow`: the force of infection in
each age group is `M(t) @ (I_w / N ** infection_pop_scale)`, where `I_w` is the weighted
infectious count in each group and `N` its population. With the exponent 1 that is frequency
dependence, with 0 density dependence. The contact rate is rescaled by the 2020 population so
that `raw_transmission_rate` keeps its order of magnitude across exponents.

summer2 spreads this over three places: the flow, the age stratification's mixing matrix and its
infectiousness adjustments. summer4 declares it as one object, `ForceOfInfection`, used as the
rate of every infection flow:

::::{grid} 1 1 2 2
:::{grid-item-card} summer2
```python
model.add_infection_generalised_flow(
    name=f"infection_from_{comp}",
    gen_infection_exp=Parameter("infection_pop_scale"),
    contact_rate=rescaled_rate * rel_sus,
    source=comp, dest="incipient")
...
age_strat.set_mixing_matrix(age_mixing_matrix)
for comp in ACTIVE_COMPS:
    age_strat.add_infectiousness_adjustments(
        comp, {age: 0. if int(age) < 15 else adj
               for age in age_groups})
```
:::
:::{grid-item-card} summer4
```python
foi = ForceOfInfection(
    "infection",
    infectious=STATE.isin(ACTIVE_COMPS),
    kind=FOIKind.GENERALISED,
    exponent=Param("infection_pop_scale"),
    contact_rate=rescaled_rate,
    infectiousness=[
        (STATE.isin(["subclin_lowinf", "clin_lowinf"]),
         Param("rel_infectiousness_lowinf")),
        (STATE.isin(["subclin_lowinf", "subclin_inf"]),
         Param("rel_infectiousness_subclin")),
        (CHILDREN, 0.0)],
    group_by=AGE,
    mixing=MixingMatrix(AGE, age_mixing_matrix,
                        normalize="none"),
)
model.add_flow(TransitionFlow(
    f"infection_from_{comp}", STATE[comp],
    STATE["incipient"], foi,
    adjust=(Multiply(rel_sus),)))
```
:::
::::

- Infectiousness weights multiply where several apply: subclinical low-infectious disease gets
  both factors, as in summer2.
- The relative susceptibility of each source state is an adjustment on its flow, not part of
  the contact rate; the force of infection is shared by the four flows.
- `normalize="none"`: the matrix is already normalised by its spectral radius (below).

## Homogeneous mixing: `group_by=None`

The original's homogeneous-mixing sensitivity analysis has no mixing matrix, so the whole
population is one mixing pool: `I_total / N_total ** exponent`. In summer4 that is
`ForceOfInfection(group_by=None)`, the default. The first port of this model, written before
summer4 0.2.0a7, needed a dummy one-trait `pool` property to group by (its workaround W1); this
port has none. `get_force_of_infection` simply leaves out `group_by` and `mixing` when there is
no matrix.

## A matrix that changes every year

The original's mixing matrix is `build_mixing_matrix(bg_mixing, a_spread, pc_strength, t)`: a
background rate, an age-assortative kernel and a parent–child term from the year's population
and fertility, normalised by its spectral radius. summer2 evaluates it inside the vector field,
with a Python loop over the 36 pairs of age bands. Everything in it depends on time only through
the calendar year, so the port builds all 186 yearly matrices at once, vectorised, from the
three parameters, and the vector field picks the current year's:

```python
stack = defer(build_mixing_stack)(Param("bg_mixing"), Param("a_spread"), Param("pc_strength"))
age_mixing_matrix = Lookup(stack, floor(Time() - 1850))
```

`stack` reads only parameters, so summer4 **hoists** it: it is computed once per run, before
the solve, and its gradient flows to the three mixing parameters. No `prepare_fn` and no
parameter-dictionary key is needed (the first port injected the stack as a parameter at run
start). The per-step work is one gather: `Lookup` takes the year's row, held at the first or last
year outside the table. (`Lookup` reads a computed table from summer4 PR #84 on; before that it
took only a parameter, and this port indexed the stack with a second `defer`.)

```{code-cell} ipython3
import numpy as np
import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"

from kiribati_tb.model import get_tb_model
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors

_, _, tv_params = get_parameters_and_priors()
tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
entries = tb.model.hoist_table.entries
stack_entries = [e for e in entries if "build_mixing_stack" in repr(e)]
print(len(entries), "parameter-only expressions are computed once per run; the mixing stack is one:")
print(repr(stack_entries[0])[:160], "...")
```

The spectral radius uses `eigvalsh` on the symmetric `diag(√N) S diag(√N)` rather than `eigvals`
on `S diag(N)`. The two have the same eigenvalues, and only the symmetric routine has reverse-mode
gradients in JAX. `tests/test_inputs.py` checks the stack against the original builder for five
parameter draws at ten times (to 1e-12) and the two radii against each other.

```{code-cell} ipython3
from kiribati_tb.age_mixing import yearly_matrices
from kiribati_tb.demographic_tools import AGE_GROUPS
from kiribati_tb.runner_tools import midpoint_params

params = midpoint_params()
stack = np.asarray(yearly_matrices(params))
labels = [f"{a}+" for a in AGE_GROUPS]
pd.DataFrame(stack[2025 - 1850], index=labels, columns=labels).round(3)
```

Rows are the infected age band and columns the infecting band, as in summer2.

## One matrix, two uses

The same `age_mixing_matrix` expression object feeds the force of infection and the computed
value `mixing_matrix_distance` (the Canberra distance to the `conmat` synthetic matrix, a
calibration target): see [step 6](06-computed-values). Because it is one object, it is evaluated
once per step for both.
