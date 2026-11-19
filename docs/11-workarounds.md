# What summer4 removed, and what remains

The first port of this model, `kiribati-tb-summer4`, was written against summer4 `c9548d5`
(after v0.2.0a5) and recorded every place it had to work around summer4 in
`docs/summer4-workarounds.md` (W1–W11) and every sharp edge it met (S1–S12). This port is
written against summer4 `main` at `5dab1b5`, the merge of PR #84 (the gaps this port found, step
52), after PR #83 (output requests and saved expressions). The tables
say what happened to each.

## Workarounds

| # | What the first port did | Status here | Why |
| --- | --- | --- | --- |
| W1 | A one-trait `pool` property so homogeneous mixing could group by something | **Dropped** | `ForceOfInfection(group_by=None)` is one mixing pool (0.2.0a7) |
| W2 | `SaveFn` leaves re-implementing the detection rates and the mixing lookup to save computed values | **Dropped** | `outputs[name] = expression` saves any rate expression (step 40, PR #83); `SaveFn` is gone |
| W3 | A `TargetSet` subclass whose `plan()` did nothing, so targets could name derived outputs | **Dropped** | `BayesianModel(outputs=)` scores targets on evaluated outputs (0.2.0a7); the stock `TargetSet` works ([step 9](09-calibration)) |
| W4 | Averted-burden quantiles computed from `PosteriorRuns.samples`, not `.differences` | **Closed in summer4** (not exercised) | `differences(..., sign=-1)` quantiles `ref − scenario` per draw (PR #84, `CX56`). This port does not do posterior full runs |
| W5 | `restrict_outputs`: an `OutputSet` with only the recorded names, every intermediate inlined | **Dropped** | `keep=False`, `kept()`, `subset(names)`; `posterior_runs` records `kept()` (step 51, PR #83) |
| W6 | A log density built with `forward_mode_differentiation=True`, bypassing `BayesianModel.potential_fn` | **Not exercised** (recorded in summer4) | `BayesianModel` still has no forward-mode option (`CX59`, `futureplans/bayesian-model-forward-mode.md`); needed only where reverse-mode gradients of the solve are NaN |
| W7 | Multi-start L-BFGS in scipy, one thread per start | **Not exercised** (partly closed in summer4) | `wf.Optax` now passes `value_fn`, so `Optax(optax.lbfgs())` runs (PR #84, `CX58`); `optimize` still vmaps every start into one program, so adaptive-solver starts run in lockstep (`CX60`) |
| W8 | Checkpointed, resumable chunked MCMC with tail-ESS stopping | **Not exercised** (partly open) | `sample_until` now keeps `extra_fields` and `StopRule(max_seconds=)` stops at a deadline; `MCMCRun` still has no save/load, and `StopRule` no tail ESS (`CX61`) |
| W9 | NUTS on a hand-sheared potential to straighten the transmission ridge | **Not exercised** (recorded in summer4) | `BayesianModel` has no hook for a joint change of coordinates (`CX62`) |
| W10 | An `MCMC` subclass to keep `num_steps` | **Closed in summer4** | `sample_until(extra_fields=...)` keeps it |
| W11 | NeuTra on a flat one-site model | **Not exercised** (recorded in summer4) | same gap as W9 (`CX62`) |

So every workaround the **model** needed (W1, W2, W3, W5) is gone, and the remaining ones are all
in the calibration *pipeline* of the first port (long MCMC runs, multi-start optimisation,
reparameterisation), which this port does not reproduce.

## Sharp edges

| # | Sharp edge | Status here |
| --- | --- | --- |
| S1 | summer4 `sharpness` defaults to 1, summer2's curvature to 16 | **Remains**: the port passes `sharpness=16.0` ([pitfalls](10-pitfalls)) |
| S2 | summer4 ran in float32 | **Gone**: summer4 enables float64 on import (0.3.0a1); the port's `jax_enable_x64` line is deleted |
| S3 | No way to tell the controller about the yearly mixing jumps | **Gone**: `Diffrax(stepsize_controller=PIDController(jump_ts=...))`; the improved variant uses it ([step 8](08-parity-and-improved)) |
| S4 | arviz before numpyro broke `numpyro.distributions` | not met (numpyro 0.19, no arviz import in the model path) |
| S5, S10, S12 | NaN reverse-mode gradients at rejected steps of the adaptive solve | not exercised (no long calibration here); the gradient at the prior midpoints is finite |
| S6 | `default_max_steps` allows 11,840 steps for 1850–2035 | **Remains**: the calibration solver sets `max_steps=4096` |
| S7–S9, S11 | NUTS warm-up, API changes, SA kernel, numpyro handlers | not exercised |

## Also removed by this port

- **`prepare_fn` for the mixing stack.** The yearly matrices are a parameter-only `defer`, which
  summer4 hoists to run start by itself ([step 4](04-mixing-and-infection)). No parameter key is
  injected.
- **A hand-written treatment-outcome expression tree.** The original's outcome function runs
  unchanged (bar `np.maximum`) through `field_refs(defer(...), NegTxOutcomes)`.
- **`_compartments`, `_flow`, `_total` helpers and the `plan` / `run` / `evaluate` boilerplate**
  in `outputs.py`: `OutputSet` requests, `by=` fan-out and `run(outputs=)` replace them.

## New gaps found

Reported to summer4 and handled in PR #84 (roadmap step 52):

| Gap | summer4 | This port |
| --- | --- | --- |
| `Lookup` took only a `Param`/`FieldRef` table, so a stack computed by a parameter-only `defer` needed a second `defer` to index it; `BayesianModel`'s docstring still pointed stacks at `prepare_fn` | **Fixed** (`CX51`): any array-valued rate expression is a table, and a parameter-only one is hoisted | `get_model_ready_age_mixing_matrix` is `Lookup(stack, floor(Time() - 1850))` ([step 4](04-mixing-and-infection)) |
| `defer` hands a per-trait argument over as a `GroupedRate`; NumPy works on it, `jnp` failed with JAX's own error | **Fixed** as a contract (`CX52`): `defer` documents it, and a `jnp` call raises an error naming the argument and the fix. JAX is not taught the type, because its operators would then drop the grouping silently | NumPy functions, as before ([pitfalls](10-pitfalls)) |
| No exact accumulator of a flow over an interval; a 0.01-year screening ramp needed the ramp times on the save grid | **Recorded** (`CX57`, `futureplans/state-ledgers-flow-integral.md`, with this port's scenario-3 numbers) | ramp times on the grid and `jump_ts`, as before |
| A fan-out's `naming` was per `OutputSet`, not per request | **Fixed** (`CX53`): `naming=` on each request, and a whole-name template | the mortality names stay a list ([step 5](05-derived-outputs)) |
| `TraitChain`'s docstring described only ageing, and its `rates` took numbers only | **Fixed** (`CX54`): documented for any pairs within a property; rates may be expressions | progression is `rates=(1 - p, p)`, not two `Dest` adjustments ([the summer4 way](13-summer4-way)) |

## Size

Lines in the two modules that changed most (total lines, and code lines without blank lines,
comments and docstrings):

| Module | summer2 original | first port | this port |
| --- | --- | --- | --- |
| `model.py` | 534 | 468 (333 code) | 580 (402 code) |
| `outputs.py` | 305 | 358 (267 code) | 266 (194 code) |
| together | 839 | 826 (600 code) | 846 (596 code) |

`outputs.py` is 27% smaller in code than the first port's and shorter than the original, though
it requests the same ~680 outputs and no longer needs a helper of its own. `model.py` is longer:
it keeps the original's five-step layout (build unstratified, stratify, then adjust flows by name),
where the first port declared the map up front and put each flow's adjustments inline. It also
holds what the first port kept elsewhere (the treatment-outcome function, the output wiring).
The layout costs about 70 lines and buys a file a summer2 user can read beside `tbh/model.py`.

The same model written compactly, `model_summer4.py` (313 lines, 256 code; 18 flows against
32), passes the same goldens: see [the summer4 way](13-summer4-way).
