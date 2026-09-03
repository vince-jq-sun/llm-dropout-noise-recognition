"""Tests for prompt-pool loading."""

from spe.prompt_utils import load_prompt_pool


def test_load_prompt_pool_is_independent_of_working_directory(
    monkeypatch,
    tmp_path,
):
    """Installed entry points can load prompt variants outside the repo root."""
    monkeypatch.chdir(tmp_path)

    pool = load_prompt_pool(
        ["classification_a_b/main_variants/v00"],
        class_names=["DROPOUT", "NOISE"],
        labels=["A", "B"],
    )

    assert len(pool) == 1
    assert pool[0]["name"] == "classification_a_b/main_variants/v00"
    assert pool[0]["turns"]
