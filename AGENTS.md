# Agent instructions

This repository is a clean summer4 port of the Kiribati TB model and an HTML guide to
converting summer2 models to summer4 (`docs/`). The first port, `kiribati-tb-summer4`, is a
separate repository; this one copies its goldens and comparison tooling.

## Rules

- **Parity is the definition of correct.** `pixi run test` must stay green against the
  committed goldens. An intended change that breaks parity goes into the `IMPROVED` variant (or a
  new one), never into `PARITY`.
- **Recognisably the original.** Keep `tbh/`'s module layout, function, parameter, flow and
  output names, typos included.
- **summer4 idiom only.** `OutputSet` requests, `outputs[name] = expr`, `field_refs(defer(...))`,
  `run(outputs=)`. No `derived_fn`, `derived_refs`, `SaveFn` or `ComputedValue` (removed).
  A new workaround of a summer4 gap gets a `# WORKAROUND(summer4):` comment and a row in
  `docs/11-workarounds.md`.
- `data/` and `tests/golden/` are inputs; goldens are regenerated only by
  `pixi run -e reference golden`, which must give identical bytes.
- Do not format `reference/tbh/` or `src/kiribati_tb/scenarios.py` (verbatim copies).
- Google Python style, line length 100, black (`pixi run format`).

## Checks

On the shared build machine every build, test and docs build goes through devq:

```bash
devq build -- pixi run format-check
devq build -- pixi run test
devq build -- pixi run docs
```
