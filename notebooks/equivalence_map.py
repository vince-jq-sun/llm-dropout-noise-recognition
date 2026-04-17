"""Build the dropout_rate to noise_std equivalence mapping.

Functions to audit sweep data, find monotonic ranges, fit isotonic
regression, and invert curves to produce matched (dropout_rate, noise_std)
pairs. Used by the map_dropout_noise notebook.
"""

import itertools
import typing

import numpy as np
import pandas as pd
import sklearn.isotonic

# -- Column name constants (match the W&B sweep DataFrames) ---------------

COL_MODEL = "model"
COL_PROMPT_TURNS = "prompt_turns"
COL_SENTENCES = "experiment.sentences_file"
CONDITION_COLS = [COL_MODEL, COL_PROMPT_TURNS, COL_SENTENCES]

COL_DROPOUT = "perturbation.dropout_rate"
COL_NOISE = "perturbation.noise_std"
COL_METRIC = "primary_log_prob_correct_mean"
COL_METRIC_SE = "primary_log_prob_correct_se"
COL_ACC = "accuracy_primary"


# -- Step 1: Audit --------------------------------------------------------


def audit_sweep(
    df: pd.DataFrame,
    perturbation_col: str,
    condition_cols: list[str] | None = None,
) -> dict[str, typing.Any]:
    """Check sweep data for duplicates and missing grid cells.

    Args:
        df: Sweep summary DataFrame (one row per run).
        perturbation_col: Column with the perturbation values.
        condition_cols: Columns that define an experimental condition.

    Returns:
        Dict with keys: n_runs, n_unique_cells, n_duplicates,
        duplicates (DataFrame or None), n_missing, missing
        (DataFrame or None), perturbation_values, n_perturbation_values.
    """
    if condition_cols is None:
        condition_cols = list(CONDITION_COLS)

    group_cols = condition_cols + [perturbation_col]
    counts = df.groupby(group_cols).size().reset_index(name="count")

    duplicates = counts[counts["count"] > 1]

    # Build the full expected grid from observed unique values
    unique_per_col = [sorted(df[col].unique()) for col in group_cols]
    full_grid = pd.DataFrame(
        list(itertools.product(*unique_per_col)),
        columns=group_cols,
    )
    merged = full_grid.merge(counts, on=group_cols, how="left")
    missing = merged[merged["count"].isna()]

    return {
        "n_runs": len(df),
        "n_unique_cells": len(counts),
        "n_duplicates": len(duplicates),
        "duplicates": duplicates if len(duplicates) > 0 else None,
        "n_missing": len(missing),
        "missing": missing if len(missing) > 0 else None,
        "perturbation_values": sorted(df[perturbation_col].unique()),
        "n_perturbation_values": df[perturbation_col].nunique(),
    }


def aggregate_sweep(
    df: pd.DataFrame,
    perturbation_col: str,
    metric_col: str = COL_METRIC,
    se_col: str = COL_METRIC_SE,
    condition_cols: list[str] | None = None,
    exclude_perturbation: list[float] | None = None,
) -> pd.DataFrame:
    """Aggregate sweep runs, collapsing duplicates by averaging.

    Uses the per-run standard error column already present in the
    W&B summary (computed over the 1000 samples within each run),
    rather than recomputing SEM across duplicate runs.

    For single-run cells (the common case), metric_mean and
    metric_se come directly from that run. For duplicate cells,
    metric_mean is the average across runs and metric_se is
    propagated via the standard formula for averaging
    (sqrt(sum(se_i^2)) / n).

    Args:
        df: Sweep summary DataFrame.
        perturbation_col: Column with the perturbation values.
        metric_col: Column with the per-run mean metric.
        se_col: Column with the per-run standard error.
        condition_cols: Columns that define an experimental condition.
        exclude_perturbation: Values to drop before aggregation
            (e.g. [0.99] for the dropout artifact).

    Returns:
        DataFrame with columns: *condition_cols, perturbation_col,
        metric_mean, metric_se, n_runs.
    """
    if condition_cols is None:
        condition_cols = list(CONDITION_COLS)

    work = df.copy()
    if exclude_perturbation:
        work = work[~work[perturbation_col].isin(exclude_perturbation)]

    group_cols = condition_cols + [perturbation_col]

    def _agg_group(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mean = g[metric_col].mean()
        if n == 1:
            se = g[se_col].iloc[0]
        else:
            # Propagate SE when averaging: SE_avg = sqrt(sum(SE_i^2)) / n
            se = float(np.sqrt((g[se_col] ** 2).sum()) / n)
        return pd.Series({"metric_mean": mean, "metric_se": se, "n_runs": n})

    agg = work.groupby(group_cols).apply(_agg_group, include_groups=False).reset_index()
    return agg


# -- Step 3: Monotonic range detection ------------------------------------


def find_monotonic_range(
    values: np.ndarray,
    performance: np.ndarray,
    dynamic_range_fraction: float = 0.05,
) -> tuple[int, int]:
    """Find the indices that bound the monotonically increasing region.

    The upper bound is the index of the peak performance. The lower
    bound is the first point whose performance exceeds the pre-peak
    minimum by a fraction of the pre-peak dynamic range.

    Note: this is an exploratory heuristic. The lower threshold uses
    only the pre-peak portion of the curve, so post-peak declines
    do not affect the lower bound. The upper bound takes the raw
    argmax, which may be a noisy spike. Visual inspection (step 2)
    is needed to validate.

    Args:
        values: Perturbation values, sorted ascending.
        performance: Mean performance at each value.
        dynamic_range_fraction: Fraction of the pre-peak dynamic
            range used to set the lower bound threshold (e.g. 0.05
            means 5% above the pre-peak minimum).

    Returns:
        (lower_index, upper_index) inclusive. The monotonic slice
        is values[lower:upper+1].
    """
    n = len(values)
    if n < 2:
        return 0, n - 1

    # Upper bound: peak
    upper_idx = int(np.argmax(performance))

    # Compute dynamic range from pre-peak portion only
    pre_peak = performance[: upper_idx + 1]
    perf_min = float(np.min(pre_peak))
    perf_max = float(pre_peak[-1])  # value at peak
    dynamic_range = perf_max - perf_min

    if dynamic_range <= 0:
        return 0, upper_idx

    # Lower bound: first pre-peak point above min + threshold
    threshold = perf_min + dynamic_range * dynamic_range_fraction
    lower_idx = 0
    for i in range(upper_idx + 1):
        if performance[i] >= threshold:
            lower_idx = i
            break

    # Ensure at least two points in the range
    if lower_idx >= upper_idx:
        lower_idx = max(0, upper_idx - 1)

    return lower_idx, upper_idx


# -- Step 4: Isotonic regression -------------------------------------------


def fit_isotonic(
    values: np.ndarray,
    performance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit increasing isotonic regression to a performance curve.

    Args:
        values: Perturbation values (used as the independent variable).
        performance: Performance at each value.

    Returns:
        (values, fitted_performance). Same values array, with
        performance replaced by the isotonic fit.
    """
    ir = sklearn.isotonic.IsotonicRegression(increasing=True)
    fitted = ir.fit_transform(values, performance)
    return np.asarray(values), np.asarray(fitted)


def deduplicate_plateau(
    values: np.ndarray,
    performance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove plateau points from an isotonic curve (left inverse).

    Where consecutive points share the same performance value, keep
    only the first (smallest perturbation value). The result is a
    strictly increasing sequence suitable for inversion.

    Args:
        values: Perturbation values, sorted ascending.
        performance: Isotonic fitted performance (non decreasing).

    Returns:
        (deduped_values, deduped_performance) with strictly
        increasing performance.
    """
    if len(performance) == 0:
        return values, performance

    mask = np.ones(len(performance), dtype=bool)
    for i in range(1, len(performance)):
        if performance[i] <= performance[i - 1]:
            mask[i] = False

    return values[mask], performance[mask]


# -- Step 5: Inversion ----------------------------------------------------


def invert_curve(
    iso_performance: np.ndarray,
    iso_values: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """Invert a strictly increasing curve using numpy.interp.

    Given a set of target performance levels, find the perturbation
    values that achieve them via linear interpolation. Values outside
    the valid range are set to NaN.

    Args:
        iso_performance: Strictly increasing performance values
            (the x axis for interpolation).
        iso_values: Corresponding perturbation values
            (the y axis for interpolation).
        target_y: Performance levels to invert.

    Returns:
        Array of matched perturbation values. NaN where the target
        falls outside the invertible range.
    """
    target_y = np.atleast_1d(np.asarray(target_y, dtype=float))
    result = np.full_like(target_y, np.nan, dtype=float)

    if len(iso_performance) < 2:
        return result

    y_min = float(iso_performance[0])
    y_max = float(iso_performance[-1])

    valid = (target_y >= y_min) & (target_y <= y_max)
    if np.any(valid):
        result[valid] = np.interp(target_y[valid], iso_performance, iso_values)

    return result


# -- Full pipeline for one condition ---------------------------------------


def build_condition_mapping(
    dropout_values: np.ndarray,
    dropout_performance: np.ndarray,
    noise_values: np.ndarray,
    noise_performance: np.ndarray,
    dynamic_range_fraction: float = 0.05,
) -> dict[str, typing.Any]:
    """Build the dropout to noise mapping for a single condition.

    Runs the full pipeline: find monotonic ranges, fit isotonic
    regression, invert the noise curve, produce matched pairs.

    Args:
        dropout_values: Sorted dropout_rate values.
        dropout_performance: Mean performance at each dropout_rate.
        noise_values: Sorted noise_std values.
        noise_performance: Mean performance at each noise_std.
        dynamic_range_fraction: Fraction for lower bound detection.

    Returns:
        Dict with keys:
        - mapping: DataFrame with columns dropout_rate,
          matched_noise_std, performance_level, in_valid_range.
        - dropout_range: (lower_idx, upper_idx) for dropout.
        - noise_range: (lower_idx, upper_idx) for noise.
        - dropout_isotonic: (values, fitted) within range.
        - noise_isotonic: (values, fitted) within range.
        - noise_deduped: (values, fitted) after deduplication.
    """
    # Find monotonic ranges
    d_lo, d_hi = find_monotonic_range(
        dropout_values,
        dropout_performance,
        dynamic_range_fraction,
    )
    n_lo, n_hi = find_monotonic_range(
        noise_values,
        noise_performance,
        dynamic_range_fraction,
    )

    # Restrict to monotonic ranges
    d_vals = dropout_values[d_lo : d_hi + 1]
    d_perf = dropout_performance[d_lo : d_hi + 1]
    n_vals = noise_values[n_lo : n_hi + 1]
    n_perf = noise_performance[n_lo : n_hi + 1]

    # Isotonic regression
    d_vals_iso, d_perf_iso = fit_isotonic(d_vals, d_perf)
    n_vals_iso, n_perf_iso = fit_isotonic(n_vals, n_perf)

    # Deduplicate noise for inversion
    n_vals_dd, n_perf_dd = deduplicate_plateau(n_vals_iso, n_perf_iso)

    # Invert
    if len(n_vals_dd) >= 2:
        matched_noise = invert_curve(n_perf_dd, n_vals_dd, d_perf_iso)
    else:
        matched_noise = np.full(len(d_vals_iso), np.nan)

    mapping_df = pd.DataFrame(
        {
            "dropout_rate": d_vals_iso,
            "matched_noise_std": matched_noise,
            "performance_level": d_perf_iso,
            "in_valid_range": ~np.isnan(matched_noise),
        }
    )

    return {
        "mapping": mapping_df,
        "dropout_range": (d_lo, d_hi),
        "noise_range": (n_lo, n_hi),
        "dropout_isotonic": (d_vals_iso, d_perf_iso),
        "noise_isotonic": (n_vals_iso, n_perf_iso),
        "noise_deduped": (n_vals_dd, n_perf_dd),
    }


# -- Build mappings for all conditions ------------------------------------


def build_all_mappings(
    df_dropout: pd.DataFrame,
    df_noise: pd.DataFrame,
    condition_cols: list[str] | None = None,
    dropout_col: str = COL_DROPOUT,
    noise_col: str = COL_NOISE,
    exclude_dropout: list[float] | None = None,
    dynamic_range_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, dict[str, typing.Any]]]:
    """Build mappings for all conditions found in the data.

    Args:
        df_dropout: Aggregated dropout sweep (from aggregate_sweep).
        df_noise: Aggregated noise sweep (from aggregate_sweep).
        condition_cols: Columns that define an experimental condition.
        dropout_col: Perturbation column in df_dropout.
        noise_col: Perturbation column in df_noise.
        exclude_dropout: Dropout values to exclude (e.g. [0.99]).
        dynamic_range_fraction: Fraction for lower bound detection.

    Returns:
        Tuple of:
        - all_mappings: DataFrame with all matched pairs across
          conditions (columns: *condition_cols, dropout_rate,
          matched_noise_std, performance_level, in_valid_range).
        - details: Dict keyed by condition tuple, with the full
          build_condition_mapping output for each condition.
    """
    if condition_cols is None:
        condition_cols = list(CONDITION_COLS)

    if exclude_dropout:
        df_dropout = df_dropout[~df_dropout[dropout_col].isin(exclude_dropout)]

    # Find conditions present in both sweeps
    dropout_conditions = set(df_dropout.groupby(condition_cols).groups.keys())
    noise_conditions = set(df_noise.groupby(condition_cols).groups.keys())
    common_conditions = sorted(dropout_conditions & noise_conditions)

    all_rows = []
    details = {}

    for condition in common_conditions:
        # Extract data for this condition
        if len(condition_cols) == 1:
            d_mask = df_dropout[condition_cols[0]] == condition
            n_mask = df_noise[condition_cols[0]] == condition
        else:
            d_mask = pd.Series(True, index=df_dropout.index)
            n_mask = pd.Series(True, index=df_noise.index)
            for col, val in zip(condition_cols, condition, strict=True):
                d_mask = d_mask & (df_dropout[col] == val)
                n_mask = n_mask & (df_noise[col] == val)

        d_sub = df_dropout[d_mask].sort_values(dropout_col)
        n_sub = df_noise[n_mask].sort_values(noise_col)

        if len(d_sub) < 2 or len(n_sub) < 2:
            continue

        d_values = d_sub[dropout_col].values
        d_perf = d_sub["metric_mean"].values
        d_se = d_sub["metric_se"].values
        n_values = n_sub[noise_col].values
        n_perf = n_sub["metric_mean"].values

        result = build_condition_mapping(
            d_values,
            d_perf,
            n_values,
            n_perf,
            dynamic_range_fraction=dynamic_range_fraction,
        )

        mapping = result["mapping"].copy()
        for col, val in zip(condition_cols, condition, strict=True):
            mapping[col] = val

        # Attach per-run SE for the dropout points in the mapping
        d_lo, d_hi = result["dropout_range"]
        mapping["dropout_se"] = d_se[d_lo : d_hi + 1]

        all_rows.append(mapping)
        details[condition] = result

    if all_rows:
        all_mappings = pd.concat(all_rows, ignore_index=True)
    else:
        all_mappings = pd.DataFrame()

    return all_mappings, details


# -- Helpers for labeling --------------------------------------------------


def sentence_label(path: str) -> str:
    """Extract a short label like '15tok' from a sentences file path.

    Args:
        path: File path like 'data/sentences/15tok.txt'.

    Returns:
        Label string like '15tok'.
    """
    import re

    match = re.search(r"(\d+tok)", str(path))
    return match.group(1) if match else str(path)


def condition_label(condition: tuple[str, ...]) -> str:
    """Build a readable label from a condition tuple.

    Args:
        condition: Tuple of (model, prompt_turns, sentences_file).

    Returns:
        Short label string.
    """
    parts = []
    for val in condition:
        s = str(val)
        # Shorten common patterns
        if "sentences/" in s:
            s = sentence_label(s)
        elif "localization_" in s:
            s = s.replace("localization_", "").replace("_2_letters", "")
        parts.append(s)
    return " | ".join(parts)
