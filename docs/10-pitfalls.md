# Pitfalls

Things that went wrong, or nearly did, converting this model. Each is easy to get wrong and easy
to fix once known.

## Numbers that change silently

**Sigmoidal interpolation curvature.** summer2's `get_sigmoidal_interpolation_function`
defaults to curvature 16; summer4's `sharpness` defaults to 1, nearly linear. Pass
`sharpness=16.0` on every sigmoidal interpolation ported from summer2. Without it the population
moved by up to 3.6 people per compartment ([step 3](03-time-varying)).

**Midpoint flow outputs.** summer2 reports a flow output as the average of its rate at
consecutive saves; summer4 reports the rate itself. Use `OutputSet(flow_values="midpoint")` for
parity, and know that the midpoint is a poor yearly count when something happens within the year
([step 8](08-parity-and-improved)).

**Cumulative outputs.** summer2's cumulative output from `start_time` includes the value
labelled `start_time`, which covers the interval before it. `cumsum(start=)` reproduces that;
`cumulative_integral(start=)` starts at 0, as the name says. Do not apply `cumulative_integral`
to midpoint values (those are amounts per interval), or `cumsum` to rates.

**Solver tolerance.** A model with a piecewise-constant input (here the yearly mixing matrix)
needs a much tighter tolerance than it looks to reach parity, or `jump_ts` on the step-size
controller. At `1e-8` both summer2gen and diffrax carried 1.5e-5 relative error from the jumps.

**summer4's default solver changed.** Since 0.3.0a2 a run without `solver=` uses adaptive Tsit5,
not Euler. Name the solver whenever numbers must match something.

## Structure

**Ageing is not automatic.** summer2's `AgeStratification` adds ageing flows; `model.stratify(AGE)`
does not. Add one `TransitionFlow("ageing", AGE.present(), AGE.present(), 1.0,
pairing=TraitChain.from_breakpoints(AGE))`.

**Name the properties for the outputs you want.** `by=` names strata `{name}X{property}_{trait}`.
The original's stratification is called `reachability` but its outputs say `reach`; calling the
summer4 property `reach` gives the original names without a naming function.

**The force of infection is one object.** summer2 attaches the mixing matrix and infectiousness
adjustments to the age stratification; summer4 declares them on `ForceOfInfection`, with the
grouping property, where the infection flows are added. A relative susceptibility belongs on each
infection flow as an adjustment, not in the shared contact rate.

**Homogeneous mixing needs no dummy property** (summer4 ≥ 0.2.0a7): `ForceOfInfection(group_by=None)`
is one mixing pool. Grouping by age with an all-ones matrix is *not* the same under a generalised
exponent: it gives per-group denominators.

**Empty aggregates.** `outputs.aggregate(name, [])` raises; summer2 returned zeros. Write
`0.0 * outputs.ref(<any output>)`.

**Keep the original's names, typos too.** `infectiousnnes_gain_clin` stays misspelled: tests,
targets and readers of both codebases look things up by name.

## Expressions

**`defer` passes summer4 values, not plain arrays.** A per-age argument arrives as a
`GroupedRate`. NumPy functions (`np.exp`, `np.maximum`) and arithmetic work on it and keep the
grouping; `jnp` functions do not, and summer4 raises a `TypeError` naming the argument. The original's `jnp.max(jnp.array((a, 0.)))` per age becomes `np.maximum(a, 0.)`.

**`Lookup` needs a parameter.** `Lookup(table, index)` gathers rows only from a `Param` or
`FieldRef`, not from a computed expression. To pick the year's matrix from a stack built by a
parameter-only `defer`, the port uses a second `defer` that indexes it
(`kiribati_tb.age_mixing.get_model_ready_age_mixing_matrix`).

**A constant is not an output.** `outputs[name] = Param("x")` raises: it is not a time series.
Multiply a saved output by it. Any expression that reads `Time()`, the state or a flow can be
saved.

**Parameter-only work is hoisted; make sure it really is parameter-only.** The yearly mixing
stack is computed once per run because it reads only `Param`s. Reading `Time()` inside it would
move it into every step.

## Environment

**Pin JAX.** summer4 sets only lower bounds. Use the 0.6 series (`jax`/`jaxlib >=0.6,<0.7`,
`numpyro <0.20`); JAX 0.12 breaks parallel NUTS chains through a diffrax solve.

**float64 is on.** Importing summer4 enables `jax_enable_x64`, as summer2gen did. Ports of
older summer4 releases turned it on themselves; that code can go.
