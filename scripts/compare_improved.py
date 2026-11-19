"""Compare the PARITY and IMPROVED variants and write the tables to ``outputs/compare/``.

Run with ``pixi run compare`` (through devq: ``devq build -- pixi run compare``).
"""

from __future__ import annotations

import pandas as pd

from kiribati_tb.comparison import (
    annual_flow_comparison,
    build,
    cumulative_comparison,
    solver_comparison,
)
from kiribati_tb.paths import OUTPUT_FOLDER
from kiribati_tb.runner_tools import midpoint_params
from kiribati_tb.variants import IMPROVED, PARITY


def main() -> None:
    out = OUTPUT_FOLDER / "compare"
    out.mkdir(parents=True, exist_ok=True)
    params = midpoint_params()
    pd.options.display.width = 160
    pd.options.display.float_format = "{:.4g}".format

    solvers = solver_comparison(build(PARITY), params)
    solvers.to_csv(out / "solvers.csv")
    print("Solvers, baseline (heterogeneous mixing), against Dopri5 at 1e-12:\n", solvers, "\n")

    for scenario in (None, "scenario_3"):
        label = scenario or "baseline"
        annual = annual_flow_comparison(build(PARITY, scenario), build(IMPROVED, scenario), params)
        annual.to_csv(out / f"annual_{label}.csv")
        print(f"Annual flows, {label}:\n", annual.loc[(slice(None), [2025.0, 2026.0, 2027.0]), :])
        summary = annual.groupby("output")[
            ["parity_err_%", "improved_rate_err_%", "improved_interval_err_%"]
        ].agg(lambda s: s.abs().max())
        summary.to_csv(out / f"annual_{label}_max_abs_err.csv")
        print(f"Largest |error| %, 2020-2035, {label}:\n", summary, "\n")

    cumulative = cumulative_comparison(build(PARITY), build(IMPROVED), params)
    cumulative.to_csv(out / "cumulative.csv")
    print("Cumulative from 2026:\n", cumulative)


if __name__ == "__main__":
    main()
