"""Classification experiment runner: multiclass (NOTHING/DROPOUT/NOISE)."""

import functools
import itertools
import random
import typing

import numpy as np
import omegaconf
import transformers

from spe import aiayn_hooks, core, generation, hooks, logit_lens
from spe.evaluation import multiclass as eval_multiclass
from spe.prompt_utils import (
    build_messages,
    build_messages_from_turns,
    compute_substring_token_range,
    print_token_map,
    validate_token_range,
)

# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------


def _build_multiclass_row(
    sample_id: int,
    perturbation: str,
    is_baseline: bool,
    permutation: tuple[str, ...],
    vdata: dict[str, typing.Any],
    generated_token_id: int,
    class_names: list[str],
    variant_token_ids: dict[str, int] | None = None,
) -> dict[str, typing.Any]:
    """Build a single result row for the multiclass experiment.

    Maps letter-level metrics (A/B/C) to class-level metrics
    (NOTHING/DROPOUT/NOISE) using the permutation.
    """
    class_to_letter = {permutation[i]: chr(65 + i) for i in range(len(permutation))}
    letter_to_class = {chr(65 + i): permutation[i] for i in range(len(permutation))}
    letters = [chr(65 + i) for i in range(len(permutation))]

    row: dict[str, typing.Any] = {
        "sample_id": sample_id,
        "perturbation": perturbation,
        "is_baseline": is_baseline,
        "option_order": "/".join(permutation),
        "generated_token_id": generated_token_id,
        "argmax_token": vdata["argmax_token"],
        "argmax_logit": vdata["argmax_logit"],
        "argmax_prob": vdata["argmax_prob"],
        "argmax_token_id": vdata["argmax_token_id"],
        "entropy": vdata["entropy"],
    }

    # Letter level metrics (position based)
    for letter in letters:
        row[f"sum_prob_{letter}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{letter}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{letter}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{letter}"] = vdata[f"primary_logit_{letter}"]
        row[f"primary_log_prob_{letter}"] = vdata[f"primary_log_prob_{letter}"]
        row[f"aggregate_log_prob_{letter}"] = vdata[f"aggregate_log_prob_{letter}"]

    # Class level metrics (semantic, mapped through the permutation)
    for cls in class_names:
        letter = class_to_letter[cls]
        row[f"sum_prob_{cls.lower()}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{cls.lower()}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{cls.lower()}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{cls.lower()}"] = vdata[f"primary_logit_{letter}"]
        row[f"primary_log_prob_{cls.lower()}"] = vdata[f"primary_log_prob_{letter}"]
        row[f"aggregate_log_prob_{cls.lower()}"] = vdata[f"aggregate_log_prob_{letter}"]

    # Logit diff: for each class X, logit(X) - logaddexp(logit(Y), logit(Z), ...)
    for cls in class_names:
        cls_lower = cls.lower()
        others_primary = [
            row[f"primary_logit_{c.lower()}"] for c in class_names if c != cls
        ]
        others_logsumexp = [
            row[f"logsumexp_{c.lower()}"] for c in class_names if c != cls
        ]
        row[f"logit_diff_{cls_lower}"] = row[
            f"primary_logit_{cls_lower}"
        ] - functools.reduce(np.logaddexp, others_primary)
        row[f"logsumexp_diff_{cls_lower}"] = row[
            f"logsumexp_{cls_lower}"
        ] - functools.reduce(np.logaddexp, others_logsumexp)

    # Pairwise logit diff: X_vs_Y = logit(X) - logit(Y)
    for i, cls_x in enumerate(class_names):
        for cls_y in class_names[i + 1 :]:
            x, y = cls_x.lower(), cls_y.lower()
            row[f"logit_diff_{x}_vs_{y}"] = (
                row[f"primary_logit_{x}"] - row[f"primary_logit_{y}"]
            )
            row[f"aggregate_logit_diff_{x}_vs_{y}"] = (
                row[f"logsumexp_{x}"] - row[f"logsumexp_{y}"]
            )

    # Predicted label (letter) and class for both aggregate and primary
    predicted_label_aggregate = max(letters, key=lambda l: row[f"sum_prob_{l}"])
    predicted_label_primary = max(letters, key=lambda l: row[f"primary_logit_{l}"])
    predicted_class_aggregate = letter_to_class[predicted_label_aggregate]
    predicted_class_primary = letter_to_class[predicted_label_primary]

    row["predicted_label_aggregate"] = predicted_label_aggregate
    row["predicted_label_primary"] = predicted_label_primary
    row["predicted_class_aggregate"] = predicted_class_aggregate
    row["predicted_class_primary"] = predicted_class_primary

    # Argmax prediction: map full-vocab argmax to a label/class (or OTHER)
    if variant_token_ids is not None:
        argmax_label = generation.argmax_to_label(
            vdata["argmax_token_id"],
            variant_token_ids,
            letters,
        )
        row["predicted_label_argmax"] = argmax_label
        row["predicted_class_argmax"] = letter_to_class.get(argmax_label, "OTHER")
    else:
        row["predicted_label_argmax"] = "OTHER"
        row["predicted_class_argmax"] = "OTHER"

    # Ground truth letter: the letter that corresponds to the ground truth class
    row["ground_truth_letter"] = class_to_letter.get(perturbation, "")

    # Log probability of the correct answer
    ground_truth_cls = perturbation.lower()
    plp_key = f"primary_log_prob_{ground_truth_cls}"
    alp_key = f"aggregate_log_prob_{ground_truth_cls}"
    if plp_key in row:
        row["primary_log_prob_correct"] = row[plp_key]
    if alp_key in row:
        row["aggregate_log_prob_correct"] = row[alp_key]

    # Per variant raw values
    for key, value in vdata.items():
        if key.startswith("v_prob_") or key.startswith("v_logit_"):
            row[key] = value

    return row


def _per_layer_logit_diff(
    lens_data: list[dict[str, float]],
    permutation: tuple[str, ...],
    labels: list[str],
    class_x: str,
    class_y: str,
) -> list[float]:
    """Compute logit(class_x) - logit(class_y) at each layer from logit lens data."""
    letter_x = labels[list(permutation).index(class_x)]
    letter_y = labels[list(permutation).index(class_y)]
    return [
        layer[f"primary_logit_{letter_x}"] - layer[f"primary_logit_{letter_y}"]
        for layer in lens_data
    ]


def run_multiclass(
    cfg: omegaconf.DictConfig,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> tuple[dict, list]:
    """Execute the multiclass (nothing/dropout/noise) experiment loop.

    Runs deterministic baselines (all permutations of the class names
    as A/B/C options, no hooks) followed by ``num_samples`` stochastic
    passes.  Samples are split evenly across the active perturbation
    types.  By default all non-NOTHING classes are active; set
    ``perturbation.active_perturbations`` to restrict sampling to a
    subset (e.g. ``["DROPOUT"]`` to generate only dropout samples).

    Args:
        cfg: Resolved Hydra config.
        model: Loaded language model.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (metrics dict, results list).
    """
    class_names = list(cfg.prompts.turns.class_names)
    labels = list(cfg.prompts.turns.labels)

    # Resolve variant token IDs for all letter variants
    all_variants = []
    for letter in labels:
        all_variants += generation.letter_variants(letter)
    variant_token_ids = generation.resolve_token_ids_safe(tokenizer, all_variants)
    if not variant_token_ids:
        raise ValueError(
            "No letter variant resolved to a single token. Cannot run multiclass mode."
        )

    print(f"\nResolved variant token IDs: {variant_token_ids}")

    # Resolve and validate layer range once (fails fast on bad config).
    resolved_first_layer, resolved_last_layer = hooks.resolve_layer_range(
        model,
        cfg.perturbation.first_layer,
        cfg.perturbation.last_layer,
    )

    # Build display name mapping from aliases config (same pattern as ICL)
    use_aliases = cfg.aliases.display_names is not None
    if use_aliases:
        display_map: dict[str, str] = dict(cfg.aliases.display_names)
        # Fill identity for classes not in alias config (e.g. NOTHING)
        for cls in class_names:
            if cls not in display_map:
                display_map[cls] = cls
        # Alias descriptions (optional, from alias config)
        alias_desc_map: dict[str, str] = {}
        if cfg.aliases.get("descriptions") is not None:
            alias_desc_map = dict(cfg.aliases.descriptions)
    else:
        display_map = {c: c for c in class_names}
        alias_desc_map = {}

    # Description pool (Part 2).  Loaded unconditionally so that
    # non-aliased classes (e.g. NOTHING) keep their descriptions
    # even when aliases are active for the other classes.
    aliased_classes: set[str] = (
        set(cfg.aliases.display_names.keys()) if use_aliases else set()
    )
    description_pool: dict[str, list[str]] | None = None
    if cfg.prompts.turns.get("description_pool") is not None:
        description_pool = {
            k: list(v) for k, v in cfg.prompts.turns.description_pool.items()
        }
        print(f"Description pool loaded: {list(description_pool.keys())}")

    # Prompt pool (Part 1): load variant configs, validate schema.
    prompt_pool: list[dict[str, typing.Any]] | None = None
    if cfg.prompts.turns.get("prompt_pool") is not None:
        from spe.prompt_utils import load_prompt_pool

        prompt_pool = load_prompt_pool(
            list(cfg.prompts.turns.prompt_pool),
            class_names=class_names,
            labels=labels,
        )

    # All 6 permutations for baseline coverage
    all_permutations = list(itertools.permutations(class_names))

    # Sentence-based mode: balanced classes with random sentences
    sentences_file = cfg.experiment.get("sentences_file", None)
    use_sentences = sentences_file is not None
    sentences_pool: list[str] = []
    if use_sentences:
        with open(sentences_file) as f:
            sentences_pool = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sentences_pool)} sentences from {sentences_file}")

    # Stochastic split
    num_samples = cfg.experiment.num_samples

    # Determine which perturbation classes to sample.
    # active_perturbations overrides the default in both sentence and
    # non-sentence mode so that independent single-perturbation sweeps
    # work regardless of how sentences are provided.
    active_raw = cfg.perturbation.get("active_perturbations", None)

    if active_raw is not None:
        active_list = list(active_raw)
        for a in active_list:
            if a not in class_names:
                raise ValueError(
                    f"active_perturbations contains '{a}' which is not in "
                    f"class_names {list(class_names)}."
                )
        active_classes = active_list
    elif use_sentences:
        # Balanced: all classes including NOTHING
        active_classes = list(class_names)
    else:
        active_classes = [c for c in class_names if c != "NOTHING"]

    samples_per_class = num_samples // len(active_classes)
    # Redistribute remainder to last class
    sample_counts: dict[str, int] = {c: samples_per_class for c in active_classes}
    sample_counts[active_classes[-1]] += num_samples - samples_per_class * len(
        active_classes
    )

    has_baselines = not use_sentences and "NOTHING" in class_names
    num_baselines = len(all_permutations) if has_baselines else 0
    total_rows = num_baselines + num_samples

    # --- Results table header ---
    print_agg = cfg.output.print_aggregates
    # Build class column headers with full names
    if print_agg:
        mc_cls_headers = [f"sumP({c.lower()})" for c in class_names]
    else:
        mc_cls_headers = [f"P({c.lower()})" for c in class_names]
    mc_cls_width = max(max(len(h) for h in mc_cls_headers), 9)
    mc_cls_cols = "  ".join(f"{h:>{mc_cls_width}}" for h in mc_cls_headers)
    table_width = (
        59 + 31 + 2 + mc_cls_width * len(class_names) + 2 * (len(class_names) - 1)
    )
    # Build letter column headers dynamically
    if print_agg:
        ltr_headers = [f"sumP({l})" for l in labels]
    else:
        ltr_headers = [f'P(" {l}")' for l in labels]
    ltr_cols = "  ".join(f"{h:>9}" for h in ltr_headers)

    counts_desc = "  ".join(f"{sample_counts[c]} {c.lower()}" for c in active_classes)
    baselines_desc = f"{num_baselines} baselines + " if has_baselines else ""

    print(f"\n{'=' * table_width}")
    print(
        f"MULTICLASS  |  {baselines_desc}{counts_desc} = {total_rows} rows  |  "
        f"dropout_rate={cfg.perturbation.dropout_rate}  |  noise_std={cfg.perturbation.noise_std}"
    )
    print(
        f"Variant IDs: {variant_token_ids}  |  Layer range: {resolved_first_layer}..{resolved_last_layer}"
    )
    print(f"{'=' * table_width}")

    def _print_table_header() -> None:
        print(
            f"{'#':>4}  {'Type':<10}  {'Order':<25}  {'Argmax':<12}  "
            f"{ltr_cols}  {mc_cls_cols}"
        )

    _print_table_header()
    print(f"{'-' * table_width}")

    results: list[dict[str, typing.Any]] = []
    num_printed_prompts = 0
    max_printed_prompts = min(3, total_rows)
    row_idx = 0

    def _make_format_kwargs(
        perm: tuple[str, ...],
        sampled_descriptions: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build format kwargs for prompt templates.

        Description precedence rule:
        - Aliased classes: label = alias name, description = alias
          config descriptions.
        - Non-aliased classes (e.g. NOTHING) and canonical mode:
          label = display_map (identity for non-aliased), description
          = sampled from description_pool or core.describe_option
          fallback.
        """
        kwargs: dict[str, str] = {}
        for i, cls in enumerate(perm):
            label = display_map[cls]
            if use_aliases and cls in aliased_classes:
                desc = alias_desc_map.get(cls, "")
            elif sampled_descriptions and cls in sampled_descriptions:
                desc = sampled_descriptions[cls]
            else:
                desc = core.describe_option(cls)
            if desc:
                kwargs[f"option_{chr(97 + i)}"] = f"{label} {desc}"
            else:
                kwargs[f"option_{chr(97 + i)}"] = label
        # Add type_a/type_b for prompts that reference perturbation names in their intro
        non_nothing = [c for c in class_names if c != "NOTHING"]
        for j, pcls in enumerate(non_nothing):
            kwargs[f"type_{chr(97 + j)}"] = display_map[pcls]
        return kwargs

    def _print_row(
        idx: int, label: str, perm: tuple[str, ...], vdata: dict, row: dict
    ) -> None:
        order_str = "/".join(perm)
        prob_prefix = "sum_prob" if print_agg else "primary_prob"
        ltr_vals = "  ".join(f"{vdata[f'{prob_prefix}_{l}']:>9.6f}" for l in labels)
        cls_vals = "  ".join(
            f"{row[f'{prob_prefix}_{c.lower()}']:>{mc_cls_width}.6f}"
            for c in class_names
        )
        print(
            f"{idx:>4}  {label:<10}  {order_str:<25}  {vdata['argmax_token']:<12}  "
            f"{ltr_vals}  {cls_vals}"
        )

    def _print_prompt_and_token_maps(
        messages: list, first_token: int, last_token: int
    ) -> None:
        print(f"\n{'=' * 60}")
        print("PROMPT (option order varies across samples)")
        print(f"{'=' * 60}")
        for msg in messages:
            print(f"[{msg['role']}]: {msg['content']}")
        print(f"{'=' * 60}")
        for ptype in class_names:
            print_token_map(
                tokenizer,
                messages,
                first_token=first_token,
                last_token=last_token,
                label=ptype,
                enable_thinking=cfg.model.thinking,
            )
        # Re-print table header after prompt block so rows align with headers
        print(f"\n{'=' * table_width}")
        _print_table_header()
        print(f"{'-' * table_width}")

    # ---- Phase 1: deterministic baselines (no hooks, only when NOTHING in classes) ----
    if has_baselines:
        for perm in all_permutations:
            format_kwargs = _make_format_kwargs(perm)
            messages = build_messages(cfg, format_kwargs=format_kwargs)
            first_token = core.resolve_first_token(cfg, tokenizer, messages)
            last_token = core.resolve_last_token(cfg, tokenizer, messages)
            validate_token_range(first_token, last_token)

            if num_printed_prompts < max_printed_prompts:
                _print_prompt_and_token_maps(messages, first_token, last_token)
                num_printed_prompts += 1

            _response, logits, generated_token_id, hidden_states = (
                core.run_single_sample(
                    model,
                    tokenizer,
                    messages,
                    hook_fn=None,
                    hook_target=cfg.perturbation.hook_target,
                    first_layer=cfg.perturbation.first_layer,
                    last_layer=cfg.perturbation.last_layer,
                    enable_thinking=cfg.model.thinking,
                )
            )

            vdata = generation.extract_n_letter_variant_data(
                logits,
                variant_token_ids,
                tokenizer,
                labels,
            )
            row = _build_multiclass_row(
                sample_id=row_idx,
                perturbation="NOTHING",
                is_baseline=True,
                permutation=perm,
                vdata=vdata,
                generated_token_id=generated_token_id,
                class_names=class_names,
                variant_token_ids=variant_token_ids,
            )
            row["logit_lens"] = logit_lens.compute_logit_lens(
                model,
                hidden_states,
                variant_token_ids,
                labels,
            )
            row["logit_lens_logit_diff_dropout_vs_noise"] = _per_layer_logit_diff(
                row["logit_lens"],
                perm,
                labels,
                "DROPOUT",
                "NOISE",
            )
            results.append(row)
            _print_row(row_idx, "BASELINE", perm, vdata, row)
            row_idx += 1

    # ---- Phase 2: stochastic passes ----
    dropout_style = cfg.perturbation.get("dropout_style", "post_sublayer")
    aiayn_positions_mc: dict[str, bool] | None = None
    if dropout_style == "aiayn":
        aiayn_positions_mc = aiayn_hooks.resolve_positions(cfg)

    # Build flat assignment list: [CLASS_A]*n_a + [CLASS_B]*n_b + ..., shuffled
    perturbation_assignments = []
    for cls in active_classes:
        perturbation_assignments += [cls] * sample_counts[cls]
    random.shuffle(perturbation_assignments)

    system_content = cfg.prompts.system.content

    for i in range(num_samples):
        perturbation_choice = perturbation_assignments[i]

        # Random permutation for option order
        perm = random.choice(all_permutations)

        # Sample descriptions (only in canonical mode with description_pool)
        sampled_descriptions: dict[str, str] | None = None
        if description_pool is not None:
            sampled_descriptions = {
                cls: random.choice(description_pool[cls])
                for cls in class_names
                if cls in description_pool
            }

        format_kwargs = _make_format_kwargs(perm, sampled_descriptions)
        sentence = ""
        if use_sentences:
            sentence = random.choice(sentences_pool)
            format_kwargs["sentence"] = sentence

        # Sample prompt variant (or use default turns)
        selected_variant: dict[str, typing.Any] | None = None
        if prompt_pool is not None:
            selected_variant = random.choice(prompt_pool)
            messages = build_messages_from_turns(
                system_content,
                selected_variant["turns"],
                format_kwargs=format_kwargs,
            )
        else:
            messages = build_messages(cfg, format_kwargs=format_kwargs)

        if use_sentences:
            first_token, last_token = compute_substring_token_range(
                tokenizer,
                messages,
                sentence,
                enable_thinking=cfg.model.thinking,
            )
        else:
            first_token = core.resolve_first_token(cfg, tokenizer, messages)
            last_token = core.resolve_last_token(cfg, tokenizer, messages)
        validate_token_range(first_token, last_token)

        if num_printed_prompts < max_printed_prompts:
            _print_prompt_and_token_maps(messages, first_token, last_token)
            num_printed_prompts += 1

        if perturbation_choice == "DROPOUT" and dropout_style == "aiayn":
            _response, logits, generated_token_id, hidden_states = (
                aiayn_hooks.run_single_sample(
                    model,
                    tokenizer,
                    messages,
                    dropout_rate=cfg.perturbation.dropout_rate,
                    first_token=first_token,
                    last_token=last_token,
                    hook_target=cfg.perturbation.hook_target,
                    first_layer=cfg.perturbation.first_layer,
                    last_layer=cfg.perturbation.last_layer,
                    aiayn_positions=aiayn_positions_mc,
                    enable_thinking=cfg.model.thinking,
                )
            )
        else:
            if perturbation_choice == "NOTHING":
                hook_fn = None
            elif perturbation_choice == "DROPOUT":
                hook_fn = hooks.create_dropout_hook(
                    cfg.perturbation.dropout_rate,
                    first_token=first_token,
                    last_token=last_token,
                )
            else:
                hook_fn = hooks.create_noise_hook(
                    cfg.perturbation.noise_std,
                    first_token=first_token,
                    last_token=last_token,
                )
            _response, logits, generated_token_id, hidden_states = (
                core.run_single_sample(
                    model,
                    tokenizer,
                    messages,
                    hook_fn=hook_fn,
                    hook_target=cfg.perturbation.hook_target,
                    first_layer=cfg.perturbation.first_layer,
                    last_layer=cfg.perturbation.last_layer,
                    enable_thinking=cfg.model.thinking,
                )
            )

        vdata = generation.extract_n_letter_variant_data(
            logits,
            variant_token_ids,
            tokenizer,
            labels,
        )
        row = _build_multiclass_row(
            sample_id=row_idx,
            perturbation=perturbation_choice,
            is_baseline=False,
            permutation=perm,
            vdata=vdata,
            generated_token_id=generated_token_id,
            class_names=class_names,
            variant_token_ids=variant_token_ids,
        )
        if use_sentences:
            row["sentence"] = sentence
        if selected_variant is not None:
            row["prompt_variant"] = selected_variant["name"]
        if sampled_descriptions is not None:
            for cls, desc in sampled_descriptions.items():
                row[f"description_{cls.lower()}"] = desc
        elif use_aliases and alias_desc_map:
            for cls, desc in alias_desc_map.items():
                row[f"description_{cls.lower()}"] = desc
        row["logit_lens"] = logit_lens.compute_logit_lens(
            model,
            hidden_states,
            variant_token_ids,
            labels,
        )
        row["logit_lens_logit_diff_dropout_vs_noise"] = _per_layer_logit_diff(
            row["logit_lens"],
            perm,
            labels,
            "DROPOUT",
            "NOISE",
        )
        results.append(row)
        _print_row(row_idx, perturbation_choice, perm, vdata, row)
        row_idx += 1

    print(f"{'=' * table_width}")

    # --- Variant summary ---
    baseline_rows = [r for r in results if r.get("is_baseline", False)]
    stochastic_classes = active_classes
    group_rows = {
        c: [
            r
            for r in results
            if r["perturbation"] == c and not r.get("is_baseline", False)
        ]
        for c in stochastic_classes
    }

    group_headers = "  ".join(f"{f'{c} P':>24}" for c in stochastic_classes)
    baseline_hdr = f"{'Baseline P':>14}  " if baseline_rows else ""
    summary_width = 12 + len(baseline_hdr) + 24 * len(stochastic_classes)

    print(f"\n{'=' * summary_width}")
    print(f"VARIANT SUMMARY (letter level, all {'/'.join(labels)} variants)")
    print(f"{'=' * summary_width}")
    print(f"{'Variant':<10}  {baseline_hdr}{group_headers}")
    print(f"{'-' * summary_width}")

    for variant in variant_token_ids:
        parts = f"{variant:<10}  "
        if baseline_rows:
            b_probs = [r.get(f"v_prob_{variant}", 0.0) for r in baseline_rows]
            parts += f"{np.mean(b_probs):>14.6f}  "
        for c in stochastic_classes:
            g_probs = [r.get(f"v_prob_{variant}", 0.0) for r in group_rows[c]]
            parts += f"{np.mean(g_probs):>10.6f} ± {np.std(g_probs):<10.6f}  "
        print(parts.rstrip())

    print(f"{'=' * summary_width}")

    metrics = eval_multiclass.compute_and_plot(
        results,
        class_names=class_names,
        labels=labels,
        dpi=cfg.output.dpi,
        fmt=cfg.output.format,
    )
    return metrics, results
