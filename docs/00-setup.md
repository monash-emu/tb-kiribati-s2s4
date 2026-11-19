# Step 0: set up, and make goldens before writing any summer4

A conversion is only as good as the check that it is right. Before porting a line, fix three
things: the environment for the original, the environment for the port, and a set of golden
outputs from the original that the port must reproduce.

## Two environments in one pixi project

`pixi.toml` has a `default` environment for the port and a `reference` environment for the
original. The reference environment reproduces the original repository's lock file exactly
(python 3.10, summer2gen `aad6d4b`, jax 0.4.38, computegraph 0.4.5, numpy 1.25.2), so the goldens
come from the stack the published analysis ran on. It never imports summer4.

```toml
[pypi-dependencies]
summer4 = { git = "https://github.com/monash-emu/summer4.git", rev = "5dab1b5a90954262bc0a27d445fe02c58b9206e0", extras = ["calibration", "pandas", "frames"] }
jax = ">=0.6,<0.7"        # summer4 develops on JAX 0.6; JAX 0.12 breaks parallel NUTS chains
jaxlib = ">=0.6,<0.7"
numpyro = ">=0.16,<0.20"  # numpyro 0.20+ needs jax 0.7

[feature.reference]
pypi-dependencies = { summerepi2 = { git = "https://github.com/monash-emu/summer2gen.git", rev = "aad6d4b414db2dbf81ea64e8aece5e0c6cdaa425" }, jax = "==0.4.38", jaxlib = "==0.4.38", computegraph = "==0.4.5" }
tasks = { golden = "python reference/make_golden.py" }
```

Pin summer4 to an exact commit or release tag, and pin JAX to the series summer4 tests on: its
`pyproject.toml` only sets lower bounds.

## Goldens

`reference/make_golden.py` builds the original model from its own source (vendored read-only in
`reference/tbh/`) and writes, for each fixture, every compartment and every derived output
(including the `save_results=False` intermediates, which it switches on) at every year:

| Fixture | Mixing | Screening | Checks |
| --- | --- | --- | --- |
| `no_transmission` | homogeneous | none | structure, ageing, the care cascade, deaths, births |
| `homog_baseline` | homogeneous | none | transmission with one mixing pool |
| `hetero_baseline` | age, yearly | none | the mixing matrix and the full model |
| `hetero_scenario_3` | age, yearly | PEARL at 84.5% | screening flows and their adjustments |
| `hetero_scenario_8` | age, yearly | CXR-TST at 84.5% | a second tool set |

It also writes the original's time-varying inputs on a half-year grid (`demography.parquet`)
and its mixing matrix for five parameter draws at ten times (`mixing.parquet`). Those let each
piece be checked on its own, before the whole model exists.

Two settings matter for goldens:

- **Tight tolerances.** The original solves at `rtol = atol = 1.4e-4`. At that tolerance its own
  numbers carry 0.4% error, which would swamp any port difference. The goldens use `1e-10`. At
  `1e-8` the yearly jumps of the mixing matrix still leave 1.5e-5 relative error, above the
  parity bar of 1e-5.
- **Every output.** Saving the intermediates too makes the name check complete: the port must
  produce exactly the original's names.

The goldens, the generator and the comparison helpers (`tests/parity.py`) in this repository are
copied from the first Kiribati port, `kiribati-tb-summer4`, unchanged.

## float64

summer2gen runs in 64-bit floats, and so does summer4: importing summer4 turns on
`jax_enable_x64`. Earlier summer4 releases did not, and ports had to; this one does nothing.
