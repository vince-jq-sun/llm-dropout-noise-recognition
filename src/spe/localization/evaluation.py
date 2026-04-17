"""Localization experiment metrics and plots (N way forced choice)."""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


def compute_and_plot(
    results: list[dict[str, typing.Any]],
    labels: list[str] | None = None,
    print_aggregates: bool = True,
    dpi: int = 300,
    fmt: str = "png",
) -> dict[str, typing.Any]:
    """Compute localization metrics, print a summary, and save plots.

    Computes and prints two versions of each classification metric:
    - **Aggregate**: predicted label is the one with the highest ``sumP``
      (from ``predicted_aggregate``).
    - **Primary**: predicted label is the one whose primary token has the
      highest logit (from ``predicted_primary``).

    When results contain perturbation type info for each position
    (as in ``dropout_vs_noise`` mode), an additional perturbation
    type analysis section is printed with a DROPOUT/NOISE confusion
    matrix and metrics.

    Args:
        results: List of result dicts, each with keys ``ground_truth``,
            ``predicted_aggregate``, ``predicted_primary``,
            ``prob_<L>``, ``logsumexp_<L>``, ``primary_prob_<L>``,
            ``primary_logit_<L>`` for each label L.
        labels: Ordered class labels (e.g. ["A", "B"] or ["A".."E"]).
            Auto detected from results if None.
        dpi: Resolution for saved plots.
        fmt: Image format (e.g. ``"png"``).

    Returns:
        Dict of computed metrics.
    """
    if labels is None:
        labels = sorted({r["ground_truth"] for r in results})

    n = len(results)
    y_true = [r["ground_truth"] for r in results]
    y_pred_aggregate = [r["predicted_aggregate"] for r in results]
    y_pred_primary = [r["predicted_primary"] for r in results]
    is_binary = len(labels) == 2

    # Detect multi-perturbation mode (e.g., DROPOUT vs NOISE)
    perturbation_keys = [f"perturbation_{lbl.lower()}" for lbl in labels]
    has_perturbation_info = bool(results) and all(
        k in results[0] for k in perturbation_keys
    )
    is_multi_perturbation = False
    if has_perturbation_info:
        ptype_set: set[str] = set()
        for r in results:
            for lbl in labels:
                ptype_set.add(r[f"perturbation_{lbl.lower()}"].upper())
        is_multi_perturbation = len(ptype_set - {"NOTHING"}) > 1

    label_str = "/".join(labels)
    print("\n" + "=" * 70)
    print(f"SUMMARY - LOCALIZATION ({label_str})")
    print("=" * 70)
    print(f"Total samples: {n}")

    # --- Distributions ---
    print("\n--- Distributions ---")
    print("\n  Ground Truth:")
    for cls in labels:
        count = sum(1 for v in y_true if v == cls)
        print(f"    {cls}: {count}")
    for y_pred, version in [
        (y_pred_aggregate, "aggregate"),
        (y_pred_primary, "primary"),
    ]:
        print(f"\n  Predictions ({version}):")
        for cls in labels:
            count = sum(1 for v in y_pred if v == cls)
            print(f"    {cls}: {count}")

    # --- Accuracy ---
    correct_agg = sum(
        1 for gt, pred in zip(y_true, y_pred_aggregate, strict=False) if gt == pred
    )
    correct_pri = sum(
        1 for gt, pred in zip(y_true, y_pred_primary, strict=False) if gt == pred
    )
    accuracy_aggregate = correct_agg / n if n > 0 else 0.0
    accuracy_primary = correct_pri / n if n > 0 else 0.0
    accuracy_se_aggregate = (
        math.sqrt(accuracy_aggregate * (1 - accuracy_aggregate) / n)
        if n > 0
        else float("nan")
    )
    accuracy_se_primary = (
        math.sqrt(accuracy_primary * (1 - accuracy_primary) / n)
        if n > 0
        else float("nan")
    )
    # Argmax accuracy: prediction based on the full-vocab argmax token
    y_pred_argmax = [r.get("predicted_argmax", "OTHER") for r in results]
    correct_argmax = sum(
        1 for gt, pred in zip(y_true, y_pred_argmax, strict=False) if gt == pred
    )
    accuracy_argmax = correct_argmax / n if n > 0 else 0.0
    accuracy_se_argmax = (
        math.sqrt(accuracy_argmax * (1 - accuracy_argmax) / n)
        if n > 0
        else float("nan")
    )
    argmax_is_label_count = sum(1 for pred in y_pred_argmax if pred != "OTHER")
    argmax_is_label_rate = argmax_is_label_count / n if n > 0 else 0.0

    print("\n--- Accuracy ---")
    print(
        f"  Aggregate: {correct_agg}/{n} = {accuracy_aggregate:.2%} "
        f"(SE: {accuracy_se_aggregate:.2%})"
    )
    print(
        f"  Primary:   {correct_pri}/{n} = {accuracy_primary:.2%} "
        f"(SE: {accuracy_se_primary:.2%})"
    )
    print(
        f"  Argmax:    {correct_argmax}/{n} = {accuracy_argmax:.2%} "
        f"(SE: {accuracy_se_argmax:.2%})"
    )
    print(
        f"  Argmax is label: {argmax_is_label_count}/{n} = {argmax_is_label_rate:.2%}"
    )

    # --- Content accuracy (control prompts with sentence_groups) ---
    content_results = [r for r in results if "content_ground_truth" in r]
    content_accuracy_aggregate: float | None = None
    content_accuracy_primary: float | None = None
    content_accuracy_se_aggregate: float | None = None
    content_accuracy_se_primary: float | None = None
    if content_results:
        n_content = len(content_results)
        content_correct_agg = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_aggregate"]
        )
        content_correct_pri = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_primary"]
        )
        content_accuracy_aggregate = content_correct_agg / n_content
        content_accuracy_primary = content_correct_pri / n_content
        content_accuracy_se_aggregate = math.sqrt(
            content_accuracy_aggregate * (1 - content_accuracy_aggregate) / n_content
        )
        content_accuracy_se_primary = math.sqrt(
            content_accuracy_primary * (1 - content_accuracy_primary) / n_content
        )
        print("\n--- Content Accuracy (control question) ---")
        print(
            f"  Aggregate: {content_correct_agg}/{n_content} = "
            f"{content_accuracy_aggregate:.2%} (SE: {content_accuracy_se_aggregate:.2%})"
        )
        print(
            f"  Primary:   {content_correct_pri}/{n_content} = "
            f"{content_accuracy_primary:.2%} (SE: {content_accuracy_se_primary:.2%})"
        )

    # --- Entropy ---
    entropy_vals = [r["entropy"] for r in results if "entropy" in r]
    entropy_mean: float | None = None
    entropy_std: float | None = None
    entropy_se: float | None = None
    if entropy_vals:
        entropy_mean = float(np.mean(entropy_vals))
        entropy_std = float(np.std(entropy_vals))
        entropy_se = entropy_std / math.sqrt(len(entropy_vals))
        print("\n--- Entropy ---")
        print(
            f"  Mean: {entropy_mean:.4f} +/- {entropy_std:.4f} (SEM: {entropy_se:.4f})"
        )

    # Per target type accuracy breakdown (when target type varies across samples)
    accuracy_by_target: dict[str, float] = {}
    target_types = {r.get("target_perturbation", "") for r in results}
    if len(target_types) > 1:
        print("\n--- Accuracy by Target Type ---")
        for ttype in sorted(target_types):
            type_results = [r for r in results if r["target_perturbation"] == ttype]
            n_type = len(type_results)
            correct_type_agg = sum(
                1 for r in type_results if r["ground_truth"] == r["predicted_aggregate"]
            )
            correct_type_pri = sum(
                1 for r in type_results if r["ground_truth"] == r["predicted_primary"]
            )
            acc_type_agg = correct_type_agg / n_type if n_type > 0 else 0.0
            acc_type_pri = correct_type_pri / n_type if n_type > 0 else 0.0
            se_type_agg = (
                math.sqrt(acc_type_agg * (1 - acc_type_agg) / n_type)
                if n_type > 0
                else float("nan")
            )
            se_type_pri = (
                math.sqrt(acc_type_pri * (1 - acc_type_pri) / n_type)
                if n_type > 0
                else float("nan")
            )
            print(f"  {ttype}:")
            print(
                f"    Aggregate: {correct_type_agg}/{n_type} = {acc_type_agg:.2%} "
                f"(SE: {se_type_agg:.2%})"
            )
            print(
                f"    Primary:   {correct_type_pri}/{n_type} = {acc_type_pri:.2%} "
                f"(SE: {se_type_pri:.2%})"
            )
            accuracy_by_target[f"accuracy_target_{ttype}_aggregate"] = acc_type_agg
            accuracy_by_target[f"accuracy_target_{ttype}_primary"] = acc_type_pri

    # --- Logit diffs (config target position vs alternatives) ---
    ld_vals = [r["logit_diff"] for r in results]
    lsd_vals = [r["logsumexp_diff"] for r in results]
    ld_mean = float(np.mean(ld_vals))
    ld_std = float(np.std(ld_vals))
    ld_se = ld_std / math.sqrt(n) if n > 0 else float("nan")
    lsd_mean = float(np.mean(lsd_vals))
    lsd_std = float(np.std(lsd_vals))
    lsd_se = lsd_std / math.sqrt(n) if n > 0 else float("nan")
    print("\n--- Logit Diffs (config target position vs alternatives) ---")
    print(f"  logit_diff:     {ld_mean:+.4f} +/- {ld_std:.4f} (SEM: {ld_se:.4f})")
    print(f"  logsumexp_diff: {lsd_mean:+.4f} +/- {lsd_std:.4f} (SEM: {lsd_se:.4f})")

    # --- Logit diffs (asked about position vs alternatives) ---
    ld_corr_vals = [r["logit_diff_correct_vs_incorrect"] for r in results]
    lsd_corr_vals = [r["logsumexp_diff_correct_vs_incorrect"] for r in results]
    ld_corr_mean = float(np.mean(ld_corr_vals))
    ld_corr_std = float(np.std(ld_corr_vals))
    ld_corr_se = ld_corr_std / math.sqrt(n) if n > 0 else float("nan")
    lsd_corr_mean = float(np.mean(lsd_corr_vals))
    lsd_corr_std = float(np.std(lsd_corr_vals))
    lsd_corr_se = lsd_corr_std / math.sqrt(n) if n > 0 else float("nan")
    # --- Log-probabilities of correct answer ---
    plp_vals = [r["primary_log_prob_correct"] for r in results]
    alp_vals = [r["aggregate_log_prob_correct"] for r in results]
    plp_mean = float(np.mean(plp_vals))
    plp_std = float(np.std(plp_vals))
    plp_se = plp_std / math.sqrt(n) if n > 0 else float("nan")
    alp_mean = float(np.mean(alp_vals))
    alp_std = float(np.std(alp_vals))
    alp_se = alp_std / math.sqrt(n) if n > 0 else float("nan")
    # --- log(1 - P(correct)) ---
    p1mp_vals = [r["primary_log_one_minus_p_correct"] for r in results]
    a1mp_vals = [r["aggregate_log_one_minus_p_correct"] for r in results]
    p1mp_mean = float(np.mean(p1mp_vals))
    p1mp_std = float(np.std(p1mp_vals))
    p1mp_se = p1mp_std / math.sqrt(n) if n > 0 else float("nan")
    a1mp_mean = float(np.mean(a1mp_vals))
    a1mp_std = float(np.std(a1mp_vals))
    a1mp_se = a1mp_std / math.sqrt(n) if n > 0 else float("nan")

    print("\n--- Log-Probability of Correct Answer ---")
    print(
        f"  primary_log_prob_correct:   {plp_mean:+.4f} +/- {plp_std:.4f} (SEM: {plp_se:.4f})"
    )
    print(
        f"  aggregate_log_prob_correct: {alp_mean:+.4f} +/- {alp_std:.4f} (SEM: {alp_se:.4f})"
    )
    print(
        f"  primary_log(1-P(correct)):  {p1mp_mean:+.4f} +/- {p1mp_std:.4f} (SEM: {p1mp_se:.4f})"
    )
    print(
        f"  aggregate_log(1-P(correct)):{a1mp_mean:+.4f} +/- {a1mp_std:.4f} (SEM: {a1mp_se:.4f})"
    )

    print("\n--- Logit Diffs (asked about position vs alternatives) ---")
    print(
        f"  logit_diff_correct_vs_incorrect:     {ld_corr_mean:+.4f} "
        f"+/- {ld_corr_std:.4f} (SEM: {ld_corr_se:.4f})"
    )
    print(
        f"  logsumexp_diff_correct_vs_incorrect: {lsd_corr_mean:+.4f} "
        f"+/- {lsd_corr_std:.4f} (SEM: {lsd_corr_se:.4f})"
    )

    # --- Confusion matrices (position level) ---
    print("\n--- Confusion Matrices ---")
    _print_confusion_matrix(y_true, y_pred_aggregate, labels, "aggregate")
    metrics_agg = _compute_classification_metrics(
        y_true,
        y_pred_aggregate,
        labels,
        "aggregate",
    )
    _print_confusion_matrix(y_true, y_pred_primary, labels, "primary")
    metrics_pri = _compute_classification_metrics(
        y_true,
        y_pred_primary,
        labels,
        "primary",
    )

    # --- Position bias ---
    print("\n--- Position Bias ---")
    position_bias_values: dict[str, float] = {}
    for lbl in labels:
        agg_vals = [r[f"prob_{lbl}"] for r in results]
        pri_vals = [r[f"primary_prob_{lbl}"] for r in results]
        mean_agg = float(np.mean(agg_vals))
        mean_pri = float(np.mean(pri_vals))
        position_bias_values[lbl] = mean_agg
        print(f'  Mean sumP({lbl}): {mean_agg:.4f}    Mean P(" {lbl}"): {mean_pri:.4f}')
    position_bias = position_bias_values[labels[0]]

    # --- ROC AUC ---
    print("\n--- ROC AUC ---")
    roc_auc_aggregate = _compute_roc(
        results,
        labels,
        is_binary,
        "prob",
        dpi,
        fmt,
        "aggregate",
    )
    roc_auc_primary = _compute_roc(
        results,
        labels,
        is_binary,
        "primary_prob",
        dpi,
        fmt,
        "primary",
    )

    # --- Plots (no terminal output) ---
    _plot_confusion_matrix(y_true, y_pred_aggregate, labels, dpi, fmt, "aggregate")
    _plot_confusion_matrix(y_true, y_pred_primary, labels, dpi, fmt, "primary")

    if is_binary:
        _plot_logit_distribution(results, labels, dpi, fmt)

    # --- Perturbation type analysis (multi-perturbation mode only) ---
    ptype_metrics: dict[str, typing.Any] = {}
    ptype_plot_files: list[str] = []
    if is_multi_perturbation:
        ptype_metrics, ptype_plot_files = _perturbation_type_analysis(
            results,
            labels,
            dpi=dpi,
            fmt=fmt,
        )

    print("=" * 70)

    # --- Collect plot files ---
    plot_files = [
        f"localization_confusion_matrix_aggregate.{fmt}",
        f"localization_confusion_matrix_primary.{fmt}",
    ]
    # Binary _plot_roc skips saving when AUC is None; N-way always saves.
    if not is_binary or roc_auc_aggregate is not None:
        plot_files.append(f"localization_roc_aggregate.{fmt}")
    if not is_binary or roc_auc_primary is not None:
        plot_files.append(f"localization_roc_primary.{fmt}")
    if is_binary:
        plot_files.append(f"localization_logit_distribution.{fmt}")
    plot_files.extend(ptype_plot_files)

    # Build return dict (always includes both aggregate and primary)
    result_metrics: dict[str, typing.Any] = {
        "_plot_files": plot_files,
        "total_samples": n,
        "accuracy_aggregate": accuracy_aggregate,
        "accuracy_primary": accuracy_primary,
        "accuracy_argmax": accuracy_argmax,
        "accuracy_se_aggregate": accuracy_se_aggregate,
        "accuracy_se_primary": accuracy_se_primary,
        "accuracy_se_argmax": accuracy_se_argmax,
        "argmax_is_label_count": argmax_is_label_count,
        "argmax_is_label_rate": argmax_is_label_rate,
        "content_accuracy_aggregate": content_accuracy_aggregate,
        "content_accuracy_primary": content_accuracy_primary,
        "content_accuracy_se_aggregate": content_accuracy_se_aggregate,
        "content_accuracy_se_primary": content_accuracy_se_primary,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_se": entropy_se,
        "position_bias": position_bias,
        "roc_auc_aggregate": roc_auc_aggregate,
        "roc_auc_primary": roc_auc_primary,
        "logit_diff_mean": ld_mean,
        "logit_diff_std": ld_std,
        "logit_diff_se": ld_se,
        "aggregate_logit_diff_mean": lsd_mean,
        "aggregate_logit_diff_std": lsd_std,
        "aggregate_logit_diff_se": lsd_se,
        "logit_diff_correct_vs_incorrect_mean": ld_corr_mean,
        "logit_diff_correct_vs_incorrect_std": ld_corr_std,
        "logit_diff_correct_vs_incorrect_se": ld_corr_se,
        "logsumexp_diff_correct_vs_incorrect_mean": lsd_corr_mean,
        "logsumexp_diff_correct_vs_incorrect_std": lsd_corr_std,
        "logsumexp_diff_correct_vs_incorrect_se": lsd_corr_se,
        "primary_log_prob_correct_mean": plp_mean,
        "primary_log_prob_correct_std": plp_std,
        "primary_log_prob_correct_se": plp_se,
        "aggregate_log_prob_correct_mean": alp_mean,
        "aggregate_log_prob_correct_std": alp_std,
        "aggregate_log_prob_correct_se": alp_se,
        "primary_log_one_minus_p_correct_mean": p1mp_mean,
        "primary_log_one_minus_p_correct_std": p1mp_std,
        "primary_log_one_minus_p_correct_se": p1mp_se,
        "aggregate_log_one_minus_p_correct_mean": a1mp_mean,
        "aggregate_log_one_minus_p_correct_std": a1mp_std,
        "aggregate_log_one_minus_p_correct_se": a1mp_se,
    }
    result_metrics.update(metrics_agg)
    result_metrics.update(metrics_pri)
    result_metrics.update(ptype_metrics)
    result_metrics.update(accuracy_by_target)

    return result_metrics


# --- Metric helpers -----------------------------------------------


def _compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    version: str,
    verbose: bool = True,
) -> dict[str, typing.Any]:
    """Compute precision, recall, F1 for a given prediction version.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Ordered label set.
        version: ``"aggregate"`` or ``"primary"``, used as key suffix.
        verbose: Whether to print the results to the terminal.

    Returns:
        Dict of ``macro_precision_{version}``, ``macro_recall_{version}``,
        ``macro_f1_{version}``.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,  # type: ignore[arg-type]
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0.0,  # type: ignore[arg-type]
    )

    if verbose:
        label_width = max(8, max(len(l) for l in labels) + 2)
        print(
            f"\n  {'Label':<{label_width}} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'Support':>8}"
        )
        print(f"  {'-' * (label_width + 42)}")
        for i, cls in enumerate(labels):
            print(
                f"  {cls:<{label_width}} {precision[i]:>10.2%} {recall[i]:>10.2%} "  # type: ignore[index]
                f"{f1[i]:>10.2%} {support[i]:>8}"  # type: ignore[index]
            )
        print(
            f"  {'Macro':<{label_width}} {macro_p:>10.2%} {macro_r:>10.2%} "
            f"{macro_f1:>10.2%}"
        )

    return {
        f"macro_precision_{version}": float(macro_p),
        f"macro_recall_{version}": float(macro_r),
        f"macro_f1_{version}": float(macro_f1),
    }


def _print_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    version: str,
) -> None:
    """Print confusion matrix grid to the terminal.

    Rows represent ground truth, columns represent predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Ordered label set.
        version: ``"aggregate"`` or ``"primary"``, used in the title.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_width = max(6, max(len(l) for l in labels) + 1)
    margin = 8  # width for "True" row label area
    row_label_width = max(len(l) for l in labels) + 1

    grid_width = col_width * len(labels)
    print(f"\n  Confusion Matrix ({version}):")
    print(f"  {'':>{margin + row_label_width}}{'Predictions':^{grid_width}}")
    print(
        f"  {'':>{margin + row_label_width}}"
        + "".join(f"{l:>{col_width}}" for l in labels)
    )
    print(f"  {'':>{margin + row_label_width}}" + "-" * grid_width)
    mid = len(labels) // 2
    for i, row_label in enumerate(labels):
        tag = "Targets" if i == mid else ""
        row_vals = "".join(f"{cm[i][j]:>{col_width}d}" for j in range(len(labels)))
        print(f"  {tag:>{margin}}{row_label:>{row_label_width}}{row_vals}")


def _compute_roc(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    is_binary: bool,
    prob_prefix: str,
    dpi: int,
    fmt: str,
    version: str,
) -> float | None:
    """Compute ROC AUC, plot curves, and return the AUC value.

    Args:
        results: Result rows.
        labels: Ordered label set.
        is_binary: Whether there are exactly 2 labels.
        prob_prefix: Column prefix (``"prob"`` for aggregate,
            ``"primary_prob"`` for primary).
        dpi: Plot resolution.
        fmt: Image format.
        version: ``"aggregate"`` or ``"primary"``.

    Returns:
        AUC value, or None if not computable.
    """
    y_true = [r["ground_truth"] for r in results]

    if is_binary:
        y_binary = [1 if gt == labels[0] else 0 for gt in y_true]
        scores = [r[f"{prob_prefix}_{labels[0]}"] for r in results]
        roc_auc_value = _compute_roc_auc(y_binary, scores)
    else:
        roc_auc_value = _compute_ovr_roc_auc(results, labels, prob_prefix)

    if roc_auc_value is not None:
        print(f"  ROC AUC ({version}): {roc_auc_value:.4f}")
    else:
        print(f"  ROC AUC ({version}): N/A (single class)")

    # Plot
    if is_binary:
        y_binary = [1 if gt == labels[0] else 0 for gt in y_true]
        scores = [r[f"{prob_prefix}_{labels[0]}"] for r in results]
        _plot_roc(y_binary, scores, roc_auc_value, dpi, fmt, version)
    else:
        _plot_ovr_roc(results, labels, prob_prefix, dpi, fmt, version)

    return roc_auc_value


def _compute_roc_auc(
    y_binary: list[int],
    scores: list[float],
) -> float | None:
    """ROC AUC, or None if only one class is present."""
    if len(set(y_binary)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_binary, scores)
    return float(auc(fpr, tpr))


def _compute_ovr_roc_auc(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    prob_prefix: str,
) -> float | None:
    """Macro one vs rest ROC AUC using probability columns."""
    y_true = [r["ground_truth"] for r in results]
    if len(set(y_true)) < 2:
        return None

    auc_values = []
    for lbl in labels:
        y_binary = [1 if gt == lbl else 0 for gt in y_true]
        if len(set(y_binary)) < 2:
            continue
        scores = [r[f"{prob_prefix}_{lbl}"] for r in results]
        fpr, tpr, _ = roc_curve(y_binary, scores)
        auc_values.append(float(auc(fpr, tpr)))

    return float(np.mean(auc_values)) if auc_values else None


def _perturbation_type_analysis(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    dpi: int = 300,
    fmt: str = "png",
) -> tuple[dict[str, typing.Any], list[str]]:
    """Compute and print perturbation type analysis for multi-perturbation localization.

    In multi-perturbation mode (e.g., DROPOUT vs NOISE), the position
    level confusion matrix (A vs B) does not reveal whether the model
    can distinguish between the two perturbation types. This function
    maps each position to its perturbation type and reports both
    aggregate (sumP) and primary token versions of:

    1. **Prediction distribution**: how often the model picked a
       position with each perturbation type, plus the accuracy (fraction
       of times the model picked the target perturbation type).

    2. **ROC AUC**: uses the probability the model assigns to each
       position as the score, and whether the position has the target
       perturbation as the binary label. This measures whether the
       probability scores discriminate between perturbation types,
       independent of the binary pick.

    3. **Mean probability by perturbation type**: the average probability
       the model assigns to positions with each perturbation type.

    4. **Perturbation type confusion matrix** (3+ types only): rows
       represent the asked perturbation type, columns represent the
       perturbation type at the predicted position. Skipped for the
       binary case (2 positions, 2 types) because the matrix is always
       symmetric by construction.

    Args:
        results: Result rows with ``perturbation_<label>`` keys.
        labels: Position labels (e.g. ``["A", "B"]``).
        dpi: Resolution for saved plots.
        fmt: Image format (e.g. ``"png"``).

    Returns:
        Tuple of (metrics dict, list of plot file names).
    """
    n = len(results)
    plot_files: list[str] = []

    # Include ALL perturbation types (including NOTHING if present)
    # so that prediction counts never miss a type.
    all_ptype_set: set[str] = set()
    for r in results:
        for lbl in labels:
            all_ptype_set.add(r[f"perturbation_{lbl.lower()}"].upper())
    ptype_labels = sorted(all_ptype_set)
    ptype_str = "/".join(ptype_labels)

    # Determine target perturbation type(s)
    target_ptypes = {r["target_perturbation"].upper() for r in results}
    if len(target_ptypes) == 1:
        target_ptype_display = next(iter(target_ptypes))
    else:
        target_ptype_display = f"RANDOMIZED ({'/'.join(sorted(target_ptypes))})"

    print(f"\n--- Perturbation Type Analysis ({ptype_str}) ---")
    print(f"  Target perturbation: {target_ptype_display}")

    ptype_metrics: dict[str, typing.Any] = {}

    for pred_key, prob_prefix, version_str in [
        ("predicted_aggregate", "prob", "aggregate"),
        ("predicted_primary", "primary_prob", "primary"),
    ]:
        # 1. Prediction distribution and accuracy
        pred_ptype_counts: dict[str, int] = dict.fromkeys(ptype_labels, 0)
        for r in results:
            pred_pos = r[pred_key]
            pred_ptype = r[f"perturbation_{pred_pos.lower()}"].upper()
            pred_ptype_counts[pred_ptype] += 1

        print(f"\n  Model picked position with ({version_str}):")
        for pt in ptype_labels:
            count = pred_ptype_counts[pt]
            pct = count / n if n > 0 else 0.0
            print(f"    {pt}: {count}/{n} = {pct:.2%}")

        ptype_correct = sum(
            1
            for r in results
            if r[f"perturbation_{r[pred_key].lower()}"].upper()
            == r["target_perturbation"].upper()
        )
        ptype_accuracy = ptype_correct / n if n > 0 else 0.0
        ptype_accuracy_se = (
            math.sqrt(ptype_accuracy * (1 - ptype_accuracy) / n)
            if n > 0
            else float("nan")
        )
        print(
            f"  Perturbation type accuracy ({version_str}): "
            f"{ptype_correct}/{n} = {ptype_accuracy:.2%} "
            f"(SE: {ptype_accuracy_se:.2%})"
        )

        ptype_metrics[f"ptype_accuracy_{version_str}"] = ptype_accuracy
        ptype_metrics[f"ptype_accuracy_se_{version_str}"] = ptype_accuracy_se
        for pt in ptype_labels:
            ptype_metrics[f"ptype_count_{pt.lower()}_{version_str}"] = (
                pred_ptype_counts[pt]
            )

        # 2. Perturbation type confusion matrix (3+ types only)
        if len(ptype_labels) > 2:
            y_true_ptype = [r["target_perturbation"].upper() for r in results]
            y_pred_ptype = [
                r[f"perturbation_{r[pred_key].lower()}"].upper() for r in results
            ]
            _print_confusion_matrix(
                y_true_ptype,
                y_pred_ptype,
                ptype_labels,
                f"ptype_{version_str}",
            )
            cm_metrics = _compute_classification_metrics(
                y_true_ptype,
                y_pred_ptype,
                ptype_labels,
                f"ptype_{version_str}",
            )
            ptype_metrics.update(cm_metrics)
            cm_filename = _plot_confusion_matrix(
                y_true_ptype,
                y_pred_ptype,
                ptype_labels,
                dpi,
                fmt,
                f"ptype_{version_str}",
            )
            if cm_filename:
                plot_files.append(cm_filename)

        # 3. ROC AUC
        if len(ptype_labels) >= 2:
            ptype_roc_y: list[int] = []
            ptype_roc_scores: list[float] = []
            for r in results:
                for lbl in labels:
                    true_pt = r[f"perturbation_{lbl.lower()}"].upper()
                    sample_target_ptype = r["target_perturbation"].upper()
                    is_target = 1 if true_pt == sample_target_ptype else 0
                    score = r[f"{prob_prefix}_{lbl}"]
                    ptype_roc_y.append(is_target)
                    ptype_roc_scores.append(score)

            ptype_roc_auc = _compute_roc_auc(ptype_roc_y, ptype_roc_scores)
            if ptype_roc_auc is not None:
                print(
                    f"  ROC AUC (perturbation type, {version_str}): {ptype_roc_auc:.4f}"
                )
                ptype_metrics[f"ptype_roc_auc_{version_str}"] = ptype_roc_auc
            else:
                print(f"  ROC AUC (perturbation type, {version_str}): N/A")

        # 4. Mean probability by perturbation type
        prob_label = "sumP" if version_str == "aggregate" else "P"
        print(f"\n  Mean probability by perturbation type ({version_str}):")
        for pt in ptype_labels:
            pt_probs: list[float] = []
            for r in results:
                for lbl in labels:
                    if r[f"perturbation_{lbl.lower()}"].upper() == pt:
                        pt_probs.append(r[f"{prob_prefix}_{lbl}"])
            mean_val = float(np.mean(pt_probs))
            std_val = float(np.std(pt_probs))
            sem_val = std_val / math.sqrt(len(pt_probs)) if pt_probs else float("nan")
            ptype_metrics[f"ptype_mean_{prob_prefix}_{pt.lower()}"] = mean_val
            ptype_metrics[f"ptype_std_{prob_prefix}_{pt.lower()}"] = std_val
            ptype_metrics[f"ptype_se_{prob_prefix}_{pt.lower()}"] = sem_val
            print(
                f"    {prob_label}({pt} positions): {mean_val:.6f} \u00b1 {std_val:.6f}"
            )

    # 5. Mean log probability by perturbation type
    for log_prob_prefix, version_str in [
        ("primary_log_prob", "primary"),
        ("aggregate_log_prob", "aggregate"),
    ]:
        print(f"\n  Mean log probability by perturbation type ({version_str}):")
        for pt in ptype_labels:
            pt_log_probs: list[float] = []
            for r in results:
                col = f"{log_prob_prefix}_{pt.lower()}"
                if col in r:
                    pt_log_probs.append(r[col])
            if not pt_log_probs:
                continue
            mean_val = float(np.mean(pt_log_probs))
            std_val = float(np.std(pt_log_probs))
            sem_val = std_val / math.sqrt(len(pt_log_probs))
            ptype_metrics[f"ptype_mean_{log_prob_prefix}_{pt.lower()}"] = mean_val
            ptype_metrics[f"ptype_std_{log_prob_prefix}_{pt.lower()}"] = std_val
            ptype_metrics[f"ptype_se_{log_prob_prefix}_{pt.lower()}"] = sem_val
            print(f"    logP({pt} positions): {mean_val:+.6f} \u00b1 {std_val:.6f}")

    # Pairwise logit diff: DROPOUT vs NOISE positions
    if results and "primary_logit_diff_dropout_vs_noise" in results[0]:
        for key_prefix in ("primary", "aggregate"):
            key = f"{key_prefix}_logit_diff_dropout_vs_noise"
            vals = [r[key] for r in results]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            se_val = std_val / math.sqrt(len(vals))
            ptype_metrics[f"mean_{key}"] = mean_val
            ptype_metrics[f"std_{key}"] = std_val
            ptype_metrics[f"se_{key}"] = se_val
            print(f"\n  {key}: {mean_val:+.4f} +/- {std_val:.4f} (SEM: {se_val:.4f})")

    return ptype_metrics, plot_files


# --- Plot helpers --------------------------------------------------


def _plot_roc(
    y_binary: list[int],
    scores: list[float],
    roc_auc_value: float | None,
    dpi: int,
    fmt: str,
    version: str,
) -> None:
    """Save ROC curve plot (binary case)."""
    if roc_auc_value is None:
        return
    fpr, tpr, _ = roc_curve(y_binary, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc_value:.4f})")
    plt.plot(
        [0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: Localization ({version})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    filename = f"localization_roc_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def _plot_ovr_roc(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    prob_prefix: str,
    dpi: int,
    fmt: str,
    version: str,
) -> None:
    """Save one vs rest ROC curves (N>2 case)."""
    y_true = [r["ground_truth"] for r in results]
    plt.figure(figsize=(8, 6))
    for lbl in labels:
        y_binary = [1 if gt == lbl else 0 for gt in y_true]
        if len(set(y_binary)) < 2:
            continue
        scores = [r[f"{prob_prefix}_{lbl}"] for r in results]
        fpr, tpr, _ = roc_curve(y_binary, scores)
        roc_auc_val = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, lw=2, label=f"{lbl} vs rest (AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Localization one vs rest ({version})")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(alpha=0.3)
    filename = f"localization_roc_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    dpi: int,
    fmt: str,
    version: str,
) -> str:
    """Save confusion matrix heatmap.

    Returns:
        The filename of the saved plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    label_str = "/".join(labels)
    plt.title(f"Confusion Matrix: Localization ({label_str}, {version})")
    filename = f"localization_confusion_matrix_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def _plot_logit_distribution(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    dpi: int,
    fmt: str,
) -> None:
    """Save histogram of logsumexp(A) - logsumexp(B) split by ground truth (binary only)."""
    l0, l1 = labels[0], labels[1]
    diff_0 = [
        r[f"logsumexp_{l0}"] - r[f"logsumexp_{l1}"]
        for r in results
        if r["ground_truth"] == l0
    ]
    diff_1 = [
        r[f"logsumexp_{l0}"] - r[f"logsumexp_{l1}"]
        for r in results
        if r["ground_truth"] == l1
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(
        diff_0,
        bins=20,
        alpha=0.6,
        label=f"GT: {l0} (target on {l0})",
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        diff_1,
        bins=20,
        alpha=0.6,
        label=f"GT: {l1} (target on {l1})",
        color="red",
        edgecolor="black",
    )
    plt.xlabel(f"logsumexp({l0}) - logsumexp({l1})")
    plt.ylabel("Count")
    plt.title("Distribution of logsumexp Difference by Ground Truth")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.savefig(f"localization_logit_distribution.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
