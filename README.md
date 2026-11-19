# kiribati-tb-summer4-clean

The Kiribati tuberculosis screening model
([`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling),
summer2gen) ported to [summer4](https://github.com/monash-emu/summer4) in summer4's unified
idiom, with an HTML guide to converting summer2 models to summer4 that uses this model as its
worked example.

- **Recognisably the same model.** `src/kiribati_tb/` has one module per original `tbh/` module
  (`model.py`, `outputs.py`, `age_mixing.py`, `demographic_tools.py`, `interventions.py`,
  `runner_tools.py`, `scenarios.py`), with the original's function, parameter, flow and output
  names.
- **Golden parity.** `PARITY` (midpoint flow outputs, Dopri5 at 1e-10) reproduces every
  compartment and every one of the ~680 derived outputs of the original, by name, for five
  fixtures (worst relative error 1.3e-7; bar 1e-5). The goldens, their generator and the
  comparison helpers come from the first port, `kiribati-tb-summer4`.
- **Improved.** `IMPROVED` (rate flow outputs, `cumulative_integral`, a 0.1-year grid plus the
  screening ramp times, Tsit5 with `jump_ts`) is a separate configuration of the same model,
  with its differences from parity measured (`docs/08`).

## Use

```bash
pixi install
pixi run test          # parity and unit tests (~40 s)
pixi run compare       # parity vs improved tables in outputs/compare/
pixi run docs          # the guide, notebooks executed, in docs/_build/html
pixi run -e reference golden   # regenerate goldens from the original summer2gen code
```

On the shared build machine run each through the queue, e.g. `devq build -- pixi run test`.

```python
from kiribati_tb.model import get_tb_model
from kiribati_tb.runner_tools import DEFAULT_MODEL_CONFIG, get_parameters_and_priors, run_model
from kiribati_tb.variants import IMPROVED, PARITY

params, priors, tv_params = get_parameters_and_priors()
tb = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params, variant=PARITY)
derived = run_model(tb, params, PARITY)          # the original's derived_outputs frame
```

## Pins

summer4 is pinned to PR #83 (`feat/save-expressions`) at `6a39451` until it merges; JAX to the
0.6 series (`jax`/`jaxlib >=0.6,<0.7`, `numpyro <0.20`), as `summer4template` does.

## Licence

BSD-2-Clause. `data/` and `reference/tbh/` are copied from `monash-emu/kiribati_tb_modelling`
(BSD-2-Clause, Copyright (c) 2025, monash-emu); see `data/README.md` and `reference/README.md`.
