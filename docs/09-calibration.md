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

# Step 9: calibration

The original calibrates with estival's `BayesianCompartmentalModel` and pymc's `DEMetropolisZ`.
summer4's `BayesianModel` is the counterpart: it holds the compiled model, the fixed parameters,
the priors and the targets, and gives numpyro a log density with gradients through the ODE
solve. The posterior is the original's; the samplers are numpyro's.

| estival (summer2) | summer4 |
| --- | --- |
| `esp.UniformPrior(name, (lo, hi))` from a table | `priors_from_frame(frame, "parameter", "distribution", "distri_param1", "distri_param2")` |
| `est.NormalTarget(output, series, stdev)` | `Target(key=output, times=, values=, likelihood=NormalLikelihood(sd=...))` |
| `get_normal_target` (sd from a % tolerance) | `NormalLikelihood.from_tolerance(values, tol_pct)` |
| a prior on a target's sd | `NormalLikelihood(sd=Uniform("mixing_dist_sd", 5, 20))` |
| `BayesianCompartmentalModel(model, params, priors, targets)` | `BayesianModel(cm, params, priors, TargetSet(targets), outputs=outputs)` |
| `bcm.loglikelihood(**params)` | `bcm.log_likelihood(params)` |
| pymc `DEMetropolisZ` | numpyro NUTS (`bcm.sample(...)`), or summer4's multi-start optimisation (`bcm.find_map(...)`) |

`kiribati_tb.calibration.get_bcm` builds it:

```python
targets = get_targets()                                  # the original's ten
outputs = tb.outputs.subset([t.key for t in targets])    # evaluate only what is scored
bcm = BayesianModel(tb.model, params, get_priors(params), TargetSet(tuple(targets)),
                    outputs=outputs, run_kwargs={"t0": 1850.0, "t1": 2035.0, "dt": 1.0,
                                                 **CALIBRATION_SOLVER})
```

Every target names a **derived output** (`tst_posXage_15+Xreach_reachable_perc`,
`notifications`, `mixing_matrix_distance`). `BayesianModel(outputs=...)` evaluates the output set
on the run's grid and scores each target on its output, so the stock `TargetSet` works. The first
port of this model needed a `TargetSet` subclass to stop summer4 from looking for target names
among the save keys (its workaround W3); summer4 0.2.0a7 fixed that.

`subset` keeps only the target outputs; summer4 evaluates those and what they read, not the
other ~600 outputs.

```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm, imported by numpyro

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from kiribati_tb.calibration import get_bcm
from kiribati_tb.runner_tools import midpoint_params

bcm, tb = get_bcm()
pd.DataFrame(
    [(t.key, t.times.tolist() if len(t.times) < 4 else f"{len(t.times)} years",
      t.values.tolist() if len(t.values) < 4 else "…") for t in bcm.targets.targets],
    columns=["output", "years", "observed"],
)
```

```{code-cell} ipython3
params = midpoint_params()
draws = {p.name: jnp.asarray(params[p.name]) for p in bcm.priors} | {"mixing_dist_sd": jnp.asarray(10.0)}
value, grad = jax.value_and_grad(bcm.log_likelihood)(draws)
print("log-likelihood at the prior midpoints:", float(value))
pd.Series({k: float(v) for k, v in grad.items()}, name="d log L / d parameter").sort_values()
```

The gradient comes through the adaptive ODE solve and the hoisted mixing-matrix build, for every
calibrated parameter at once.

## Solver for calibration

`CALIBRATION_SOLVER` is the original's: Dopri5 at summer2gen's default `rtol = atol = 1.4e-4`,
on the parity outputs. [Step 8](08-parity-and-improved) shows that setting carries up to 0.4%
error in compartments, and that the improved solver (`CALIBRATION_IMPROVED`) is twenty times more
accurate than summer4's default and 200 times more than summer2's, for 1.5 times the steps of the
latter. Use it with `get_bcm(run_kwargs=CALIBRATION_IMPROVED)`.

The first Kiribati port (`kiribati-tb-summer4`) has a full sampling pipeline: Latin-hypercube
design, multi-start L-BFGS, a Laplace metric, sheared coordinates for the transmission ridge,
checkpointed NUTS until R-hat and ESS targets are met, and long reference runs. Those pieces work
around summer4 calibration gaps that are still open (its W6–W11, listed on
[the workarounds page](11-workarounds)) and are not repeated here.

## Sensitivity analyses

The original's three sensitivity analyses map onto the same pieces:

- `tpt_60`: `params["tpt_completion_perc"] = 60.0` before `get_bcm(params)`;
- `subclinical_50`: replace that target's observation with 50%;
- `homogeneous_mixing`: `get_bcm(heterogeneous_mixing=False)`, which drops the mixing priors and
  the distance target and uses `ForceOfInfection(group_by=None)`.
