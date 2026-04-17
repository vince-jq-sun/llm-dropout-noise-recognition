"""Tests for compute_substring_token_range (no GPU required)."""

import pytest
from transformers import AutoTokenizer

from spe.prompt_utils import compute_substring_token_range

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _make_messages(user_text: str) -> list[dict[str, str]]:
    """Build a minimal chat with one user turn."""
    return [{"role": "user", "content": user_text}]


def _decode_range(tokenizer, messages, first, last, enable_thinking=False):
    """Decode the token range [first, last] from the full prompt."""
    is_prefill = messages[-1]["role"] == "assistant"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return tokenizer.decode(ids[first : last + 1])


# --- Tests ---


def test_basic_single_occurrence(tokenizer):
    """Substring appears once — returns a valid range."""
    messages = _make_messages("The quick brown fox jumps over the lazy dog.")
    first, last = compute_substring_token_range(
        tokenizer,
        messages,
        "brown fox jumps",
    )
    assert first <= last
    decoded = _decode_range(tokenizer, messages, first, last)
    assert "brown fox jumps" in decoded


def test_anchored_disambiguation(tokenizer):
    """Substring appears twice — anchor selects the right one."""
    text = "Sentence A: The cat sat. Sentence B: The cat sat."
    messages = _make_messages(text)

    # Without anchor → first occurrence
    first1, _ = compute_substring_token_range(
        tokenizer,
        messages,
        "The cat sat.",
    )
    # With anchor → second occurrence
    first2, last2 = compute_substring_token_range(
        tokenizer,
        messages,
        "The cat sat.",
        anchor="Sentence B: ",
    )
    assert first2 > first1, "Anchored match should be later in the sequence"

    decoded = _decode_range(tokenizer, messages, first2, last2)
    assert "The cat sat." in decoded


def test_substring_not_found(tokenizer):
    messages = _make_messages("Hello world.")
    with pytest.raises(ValueError, match="not found"):
        compute_substring_token_range(tokenizer, messages, "goodbye universe")


def test_anchor_not_found(tokenizer):
    messages = _make_messages("Hello world.")
    with pytest.raises(ValueError, match="not found"):
        compute_substring_token_range(
            tokenizer,
            messages,
            "world",
            anchor="WRONG: ",
        )


def test_token_boundary_alignment(tokenizer):
    """Substring that starts/ends mid-token still returns a covering range."""
    # "unbelievable" likely tokenizes into multiple subwords;
    # asking for a slice that cuts across token boundaries.
    messages = _make_messages("That was unbelievable and extraordinary.")
    first, last = compute_substring_token_range(
        tokenizer,
        messages,
        "unbelievable",
    )
    assert first <= last
    decoded = _decode_range(tokenizer, messages, first, last)
    assert "unbelievable" in decoded


def test_roundtrip_coverage(tokenizer):
    """Decoded token range fully covers the original substring."""
    substrings = [
        "42 is the answer",
        "x = f(y)",
        "multi-word hyphenated-expression here",
    ]
    for sub in substrings:
        messages = _make_messages(f"Prefix text. {sub}. Suffix text.")
        first, last = compute_substring_token_range(
            tokenizer,
            messages,
            sub,
        )
        decoded = _decode_range(tokenizer, messages, first, last)
        assert sub in decoded, f"'{sub}' not found in decoded '{decoded}'"
