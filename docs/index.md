# Converting a summer2 model to summer4: the Kiribati TB model

This guide converts a real, published summer2 model to
[summer4](https://github.com/monash-emu/summer4), step by step, and checks every step against
the original's numbers. The model is the Kiribati tuberculosis screening model
([`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling),
built on summer2gen): 10 TB states × 8 age bands × 2 reachability strata = 160 compartments,
1850–2035, UN demography, a time-varying age-mixing matrix, a generalised force of infection,
screening scenarios, about 680 derived outputs, and a Bayesian calibration.

Read it with the two codebases side by side. The summer4 package `kiribati_tb` has one module
per original `tbh` module, with the original's function names, parameter names, flow names and
output names:

| `tbh` (summer2) | `kiribati_tb` (summer4) | What it holds |
| --- | --- | --- |
| `demographic_tools.py` | `demographic_tools.py` | UN population and mortality, prepared once in pandas |
| `age_mixing.py` | `age_mixing.py` | the yearly age-mixing matrix and its distance to `conmat` |
| `model.py` | `model.py` | `get_tb_model`, `get_natural_tb_model`, `add_detection_and_treatment`, `stratify_model_by_age`, `stratify_model_by_reachability`, `add_births_and_deaths` |
| `outputs.py` | `outputs.py` | `request_model_outputs`, `request_per_capita_output` |
| `interventions.py` | `interventions.py` | `ScreeningTools`, `ScreeningProgram`, `Scenario` |
| `data/scenarios.py` | `scenarios.py` | the screening scenarios, unchanged but for one import |
| `runner_tools.py` | `runner_tools.py`, `calibration.py` | config, parameters, running; the estival calibration |
| — | `model_summer4.py` | the same model in compact summer4 idiom ([the summer4 way](13-summer4-way)) |
| — | `model_strata.py` | the same model with TB disease stratified by clinical status and infectiousness ([two properties](14-disease-strata)) |
| — | `variants.py` | the `PARITY` and `IMPROVED` ways to solve and report |

**Correct means parity.** The original was run once, in its own pinned environment, to make
*golden* outputs (`tests/golden/`). The `PARITY` configuration reproduces every compartment and
every derived output, by name, to within 1e-5 relative (in practice 1.3e-7) for five fixtures.
A separate `IMPROVED` configuration then shows what summer4 does better, with the differences
measured.

## The steps

```{toctree}
:maxdepth: 1
:numbered:

00-setup
01-structure
02-flows-and-rates
03-time-varying
04-mixing-and-infection
05-derived-outputs
06-computed-values
07-validation
13-summer4-way
14-disease-strata
08-parity-and-improved
09-calibration
10-pitfalls
11-workarounds
12-api
```

## In one table

| summer2 | summer4 | Page |
| --- | --- | --- |
| `CompartmentalModel(compartments=...)` | `FlowModel(PropertyMap.from_property(Property("state", ...)))` | [Structure](01-structure) |
| `AgeStratification` / `Stratification` + `stratify_with` | `model.stratify(AGE)` + an ageing flow with `TraitChain` | [Structure](01-structure) |
| `set_flow_adjustments(flow, {stratum: value})` | `model.adjust_flow(flow, Multiply(value, where=AGE["15"]))` | [Structure](01-structure) |
| `source_strata=` / `dest_strata=` | selectors: `STATE["clin_inf"] & AGE["0"]` | [Flows](02-flows-and-rates) |
| `Parameter("x")`, `Time`, `Function(f, [...])` | `Param("x")`, `Time()`, arithmetic, `defer(f)(...)` | [Time-varying](03-time-varying) |
| `get_sigmoidal_interpolation_function(x, y)` | `Data(x, y).interp("sigmoidal", sharpness=16)` | [Time-varying](03-time-varying) |
| `add_infection_generalised_flow` + `set_mixing_matrix` | `ForceOfInfection(kind=GENERALISED, mixing=MixingMatrix(...))` | [Mixing](04-mixing-and-infection) |
| `request_output_for_*`, `request_aggregate_output` | `OutputSet.compartments`, `.flow`, `.aggregate`, `by=` | [Outputs](05-derived-outputs) |
| `add_computed_value_func` + `request_computed_value_output` | `outputs[name] = <rate expression>` | [Computed values](06-computed-values) |
| `model.run(params)`; `get_derived_outputs_df()` | `cm.run(params, ..., outputs=outputs)` | [Outputs](05-derived-outputs) |
| one flow per compartment with the same rate | one flow over a selector; `TraitChain(STATE, pairs)` for paired states | [The summer4 way](13-summer4-way) |
| estival `BayesianCompartmentalModel` | `BayesianModel(cm, params, priors, TargetSet(...), outputs=...)` | [Calibration](09-calibration) |

## Reproducing

```bash
pixi run test                     # golden parity and unit tests
pixi run compare                  # parity vs improved tables in outputs/compare/
pixi run docs                     # this site, notebooks executed, in docs/_build/html
pixi run -e reference golden      # regenerate the goldens from the original summer2gen code
```

On the shared build machine, run each through the queue: `devq build -- pixi run test`.
