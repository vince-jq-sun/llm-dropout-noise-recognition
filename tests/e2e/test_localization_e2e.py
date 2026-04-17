"""E2E tests for ``spe-localize`` (localization / N-way forced choice) entrypoint.

Tests the full pipeline: config -> sentence sampling -> prompt construction ->
multi-perturbation hook creation -> (mocked) generation -> per-label metric
extraction -> result building, without loading a real model.
"""

import pytest
from omegaconf import OmegaConf

from spe.experiments.localization import run_localization

# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The cat sat quietly on the warm mat",
    "A large dog ran through the green park",
    "She walked along the beautiful sandy beach",
    "The old library had many interesting books",
    "He drove his car through the busy streets",
]


def _cfg(
    num_samples: int = 4,
    num_sentences: int = 2,
    target_perturbation: str = "DROPOUT",
    num_bg_dropout: int = 0,
    num_bg_noise: int = 0,
    num_bg_nothing: int = 1,
    same_sentence: bool = True,
    seed: int = 42,
):
    """Build a minimal resolved localization config."""
    return OmegaConf.create(
        {
            "prompts": {
                "system": {"content": ""},
                "turns": {
                    "name": "test",
                    "prompt_pool": None,
                    "sentences": _SENTENCES,
                    "turns": [
                        {
                            "role": "user",
                            "content": "Detect which sentence had "
                            "{target_perturbation_name}.",
                        },
                        {"role": "assistant", "content": "I understand."},
                        {"role": "user", "content": "{sentences_block}"},
                        {"role": "assistant", "content": "Introspecting."},
                        {
                            "role": "user",
                            "content": "Which sentence? {answer_instruction}",
                        },
                        {"role": "assistant", "content": "the answer is:"},
                    ],
                },
            },
            "model": {"thinking": False},
            "perturbation": {
                "mode": "localization",
                "target_perturbation": target_perturbation,
                "num_sentences": num_sentences,
                "num_bg_dropout": num_bg_dropout,
                "num_bg_noise": num_bg_noise,
                "num_bg_nothing": num_bg_nothing,
                "dropout_rate": 0.1,
                "dropout_style": "post_sublayer",
                "noise_std": 0.01,
                "hook_target": "both",
                "first_layer": 0,
                "last_layer": -1,
                "first_token": 0,
                "last_token": -1,
                "same_sentence": same_sentence,
            },
            "output": {"print_aggregates": False, "dpi": 72, "format": "png"},
            "experiment": {
                "seed": seed,
                "num_samples": num_samples,
                "sentences_file": None,
                "sentence_n_tokens": None,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs(localization_mocks, tokenizer, fake_model):
    """Smoke test: full localization pipeline completes with the right
    number of result rows."""
    cfg = _cfg(num_samples=4)
    metrics, results = run_localization(cfg, fake_model, tokenizer)

    assert isinstance(metrics, dict)
    assert isinstance(results, list)
    assert len(results) == 4


def test_result_row_structure(localization_mocks, tokenizer, fake_model):
    """Each result row contains all expected label-level and aggregate
    metric keys."""
    cfg = _cfg(num_samples=2)
    _, results = run_localization(cfg, fake_model, tokenizer)
    row = results[0]

    # Bookkeeping
    for key in (
        "sample_id",
        "ground_truth",
        "predicted",
        "predicted_aggregate",
        "predicted_primary",
        "predicted_argmax",
        "entropy",
        "same_sentence",
        "target_perturbation",
        "first_token",
        "last_token",
    ):
        assert key in row, f"Missing bookkeeping key: {key}"

    # Per-label metrics (A, B for 2 sentences)
    for label in ("A", "B"):
        assert f"prob_{label}" in row
        assert f"logsumexp_{label}" in row
        assert f"primary_prob_{label}" in row
        assert f"primary_logit_{label}" in row
        assert f"primary_log_prob_{label}" in row
        assert f"aggregate_log_prob_{label}" in row

    # Sentence text and perturbation assignment
    for label in ("a", "b"):
        assert f"sentence_{label}" in row
        assert f"perturbation_{label}" in row
        assert f"n_perturbed_tokens_{label}" in row

    # Logit diffs
    assert "logit_diff" in row
    assert "logsumexp_diff" in row
    assert "logit_diff_correct_vs_incorrect" in row
    assert "logsumexp_diff_correct_vs_incorrect" in row

    # Log prob of correct answer
    assert "primary_log_prob_correct" in row
    assert "aggregate_log_prob_correct" in row


def test_ground_truth_position_varies(
    localization_mocks,
    tokenizer,
    fake_model,
):
    """Target position is randomized across samples."""
    cfg = _cfg(num_samples=20)
    _, results = run_localization(cfg, fake_model, tokenizer)

    ground_truths = {r["ground_truth"] for r in results}
    assert len(ground_truths) > 1, "Expected target position to vary"


def test_same_sentence_mode(localization_mocks, tokenizer, fake_model):
    """When ``same_sentence=True``, all sentences in a sample are identical."""
    cfg = _cfg(num_samples=4, same_sentence=True)
    _, results = run_localization(cfg, fake_model, tokenizer)

    for r in results:
        assert r["sentence_a"] == r["sentence_b"]


def test_different_sentences_mode(localization_mocks, tokenizer, fake_model):
    """When ``same_sentence=False``, sentences in a sample differ."""
    cfg = _cfg(num_samples=4, same_sentence=False)
    _, results = run_localization(cfg, fake_model, tokenizer)

    for r in results:
        assert r["sentence_a"] != r["sentence_b"]


def test_invalid_background_counts_raises(tokenizer, fake_model):
    """Background perturbation counts must sum to ``num_sentences - 1``."""
    cfg = _cfg(
        num_bg_dropout=1,
        num_bg_noise=1,
        num_bg_nothing=0,
    )  # sums to 2, but num_sentences - 1 = 1
    with pytest.raises(ValueError, match="Background counts must sum"):
        run_localization(cfg, fake_model, tokenizer)


def test_target_in_background_raises(tokenizer, fake_model):
    """Target perturbation type must not appear in the background."""
    cfg = _cfg(
        target_perturbation="DROPOUT",
        num_bg_dropout=1,
        num_bg_nothing=0,
    )
    with pytest.raises(ValueError, match="must not appear in background"):
        run_localization(cfg, fake_model, tokenizer)
