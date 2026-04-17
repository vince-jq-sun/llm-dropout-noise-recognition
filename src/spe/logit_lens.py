"""Logit lens: project intermediate hidden states through the model's unembedding head."""

import torch
import torch.nn.functional as F
import transformers


def _get_norm_and_head(
    model: transformers.AutoModelForCausalLM,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Locate the final layer norm and lm_head in a HuggingFace causal LM."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):  # type: ignore[attr-defined]
        return model.model.norm, model.lm_head  # type: ignore[attr-defined]
    raise ValueError("Could not find norm/lm_head in model")


def compute_logit_lens(
    model: transformers.AutoModelForCausalLM,
    hidden_states: tuple[torch.Tensor, ...],
    variant_token_ids: dict[str, int],
    letters: list[str],
) -> list[dict[str, float]]:
    """Compute per-layer logits and probabilities for letter variants.

    Args:
        model: The causal LM (used to access norm and lm_head).
        hidden_states: Tuple of ``num_layers + 1`` tensors of shape ``[hidden_dim]``
            (last token position; index 0 is the embedding output).
        variant_token_ids: Mapping from variant string to vocab index.
        letters: Ordered list of uppercase letters (e.g. ``["A", "B", "C"]``).

    Returns:
        List of dicts (one per layer), each containing:
        - ``primary_logit_<L>`` and ``primary_prob_<L>`` for each letter L
    """
    norm, lm_head = _get_norm_and_head(model)

    primary_tids = {}
    for letter in letters:
        primary = f" {letter}"
        if primary in variant_token_ids:
            primary_tids[letter] = variant_token_ids[primary]

    results = []
    with torch.no_grad():
        for hs in hidden_states:
            logits = lm_head(norm(hs)).float()  # [vocab_size]
            probs = F.softmax(logits, dim=-1)

            layer_data: dict[str, float] = {}
            for letter in letters:
                if letter in primary_tids:
                    tid = primary_tids[letter]
                    layer_data[f"primary_logit_{letter}"] = logits[tid].item()
                    layer_data[f"primary_prob_{letter}"] = probs[tid].item()
                else:
                    layer_data[f"primary_logit_{letter}"] = float("-inf")
                    layer_data[f"primary_prob_{letter}"] = 0.0

            results.append(layer_data)

    return results
