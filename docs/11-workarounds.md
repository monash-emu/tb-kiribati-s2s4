# What summer4 removed, and what remains

The first port of this model, `kiribati-tb-summer4`, was written against summer4 `c9548d5`
(after v0.2.0a5) and recorded every place it had to work around summer4 in
`docs/summer4-workarounds.md` (W1–W11) and every sharp edge it met (S1–S12). This port is
written against summer4 PR #83 (`6a39451`, output requests and saved expressions). The tables
say what happened to each.

## Workarounds

| # | What the first port did | Status here | Why |
| --- | --- | --- | --- |
| W1 | A one-trait `pool` property so homogeneous mixing could group by something | **Dropped** | `ForceOfInfection(group_by=None)` is one mixing pool (0.2.0a7) |
| W2 | `SaveFn` leaves re-implementing the detection rates and the mixing lookup to save computed values | **Dropped** | `outputs[name] = expression` saves any rate expression (step 40, PR #83); `SaveFn` is gone |
| W3 | A `TargetSet` subclass whose `plan()` did nothing, so targets could name derived outputs | **Dropped** | `BayesianModel(outputs=)` scores targets on evaluated outputs (0.2.0a7); the stock `TargetSet` works ([step 9](09-calibration)) |
| W4 | Averted-burden quantiles computed from `PosteriorRuns.samples`, not `.differences` | **Not exercised** (open in summer4) | `differences` still gives quantiles of `scenario − ref`; the original reports `ref − scenario`, and quantiles do not commute with negation. This port does not do posterior full runs |
| W5 | `restrict_outputs`: an `OutputSet` with only the recorded names, every intermediate inlined | **Dropped** | `keep=False`, `kept()`, `subset(names)`; `posterior_runs` records `kept()` (step 51, PR #83) |
| W6 | A log density built with `forward_mode_differentiation=True`, bypassing `BayesianModel.potential_fn` | **Not exercised** (open in summer4) | `BayesianModel` still has no forward-mode option; needed only where reverse-mode gradients of the solve are NaN |
| W7 | Multi-start L-BFGS in scipy, one thread per start | **Not exercised** (open in summer4) | `optimize` still vmaps every start into one program, so adaptive-solver starts run in lockstep |
| W8 | Checkpointed, resumable chunked MCMC with tail-ESS stopping | **Not exercised** (partly open) | `sample_until` now keeps `extra_fields`; `MCMCRun` still has no save/load, and `StopRule` no tail ESS |
| W9 | NUTS on a hand-sheared potential to straighten the transmission ridge | **Not exercised** (open in summer4) | `BayesianModel` has no hook for a joint change of coordinates |
| W10 | An `MCMC` subclass to keep `num_steps` | **Closed in summer4** | `sample_until(extra_fields=...)` keeps it |
| W11 | NeuTra on a flat one-site model | **Not exercised** (open in summer4) | same gap as W9 |

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

Recorded for summer4, not fixed here:

- `Lookup` accepts only a `Param`/`FieldRef` table, so a table computed by a parameter-only
  `defer` (hoisted, differentiable) cannot be gathered with it; a second `defer` that indexes it
  works. `BayesianModel`'s docstring still points run-start work such as yearly mixing stacks at
  `compile(prepare_fn=...)`, which a hoisted `defer` now covers.
- `defer` hands a per-trait argument to the function as a `GroupedRate`, which supports NumPy
  ufuncs but not `jnp` functions. Nothing in the guide says so; the error is JAX's
  "not a valid JAX type".
- summer4 has no exact accumulator of a flow over an interval, so yearly counts are only as good
  as the save grid. A 0.01-year screening ramp needed the ramp times on the grid
  (`futureplans/state-ledgers-flow-integral.md` plans one).
- A fan-out's `naming` is set per `OutputSet`, not per request, so outputs named in a different
  pattern (the original's `all_cause_mortality_from_{comp}_age_{age}`) need a loop.

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
