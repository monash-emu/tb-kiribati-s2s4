"""Configuration, parameters and running (the non-estival half of ``tbh/runner_tools.py``).

summer2's ``model.run(params)`` followed by ``model.get_outputs_df()`` and
``model.get_derived_outputs_df()`` becomes :func:`run_model`: one ``run(outputs=...)`` call, with
the variant's solver and save grid, returning the same two frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from summer4 import Compartments, OutputSet, Result, SavePlan

from kiribati_tb.model import TBModel
from kiribati_tb.paths import DATA_FOLDER
from kiribati_tb.variants import END_TIME, PARITY, START_TIME, Variant

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "start_time": int(START_TIME),
    "end_time": int(END_TIME),
    "seed": 100,
    "iso3": "KIR",
    "age_groups": ["0", "3", "5", "10", "15", "18", "40", "65"],
    # 40,483 people in South Tarawa in 2023 (excluding Betio, Bairiki and Nanikai) out of a
    # Kiribati population of 119,438.
    "pop_scaling": 40483.0 / 119438.0,
    "heterogeneous_mixing": True,
}


@cache
def _read_sheets() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = DATA_FOLDER / "parameters.xlsx"
    return (
        pd.read_excel(path, sheet_name="constant"),
        pd.read_excel(path, sheet_name="time_variant", index_col=0),
    )


def get_parameters_and_priors() -> tuple[dict[str, float], pd.DataFrame, dict[str, pd.Series]]:
    """Read ``parameters.xlsx`` as the original does, without estival.

    Returns:
        ``(params, priors, tv_params)``: every constant parameter's value; the rows that carry
        a prior (``parameter``, ``distribution``, ``distri_param1``, ``distri_param2``); and
        each time-variant column as a series indexed by year.
    """
    constant, time_variant = _read_sheets()
    params = {str(k): float(v) for k, v in zip(constant["parameter"], constant["value"])}
    priors = constant[constant["distribution"].notna()][
        ["parameter", "distribution", "distri_param1", "distri_param2"]
    ].reset_index(drop=True)
    tv_params = {str(col): time_variant[col].dropna() for col in time_variant.columns}
    return params, priors, tv_params


def midpoint_params() -> dict[str, float]:
    """Constant parameters with every uniform prior at the midpoint of its range."""
    params, priors, _ = get_parameters_and_priors()
    for row in priors.itertuples(index=False):
        if row.distribution != "uniform":
            raise ValueError(f"Unsupported prior {row.distribution!r}")
        params[str(row.parameter)] = 0.5 * (float(row.distri_param1) + float(row.distri_param2))
    return params


def _frame(named: Result, index: np.ndarray) -> pd.DataFrame:
    frame = named.to_frame(shape="wide", backend="pandas")
    frame.index = pd.Index(np.asarray(index, dtype=float), name="time")
    return frame


def run_model(
    tb_model: TBModel,
    params: Mapping[str, Any],
    variant: Variant = PARITY,
    *,
    outputs: OutputSet | None = None,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Solve 1850–2035 and return the derived outputs (``get_derived_outputs_df``).

    Args:
        tb_model: From :func:`kiribati_tb.model.get_tb_model`.
        params: Parameter values.
        variant: Solver and save grid. Its flow-output convention was fixed when the model's
            outputs were built, so pass the same variant to both.
        outputs: Another ``OutputSet`` to evaluate instead of ``tb_model.outputs``.
        names: Which outputs to return; the kept outputs by default. Pass
            ``list(outputs.keys())`` to include the ``keep=False`` intermediates.
    """
    outputs = tb_model.outputs if outputs is None else outputs
    times = variant.save_times(tb_model.breakpoints)
    plan = outputs.plan(SavePlan(ts=times))
    result = tb_model.model.run(
        params,
        t0=START_TIME,
        t1=END_TIME,
        dt=variant.save_step,
        save=plan,
        **variant.run_kwargs(tb_model.breakpoints),
    )
    named = outputs.evaluate(result, params, names=names)
    return _frame(named, times)


def compartment_labels(tb_model: TBModel) -> list[str]:
    """summer2 compartment names (``{state}Xage_{age}Xreachability_{reach}``) in map order."""
    rows = tb_model.model.pmap.to_dicts()
    return [f"{r['state']}Xage_{r['age']}Xreachability_{r['reach']}" for r in rows]


def run_compartments(
    tb_model: TBModel, params: Mapping[str, Any], variant: Variant = PARITY
) -> pd.DataFrame:
    """Compartment sizes at the save times, by summer2 compartment name (``get_outputs_df``)."""
    outputs = OutputSet()
    outputs["compartments"] = Compartments()
    times = variant.save_times(tb_model.breakpoints)
    result = tb_model.model.run(
        params,
        t0=START_TIME,
        t1=END_TIME,
        dt=variant.save_step,
        save=outputs.plan(SavePlan(ts=times)),
        **variant.run_kwargs(tb_model.breakpoints),
    )
    values = np.asarray(result["compartments"].values.data)
    frame = pd.DataFrame(values, columns=compartment_labels(tb_model))
    frame.index = pd.Index(times, name="time")
    return frame
