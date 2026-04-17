"""Shared experiment primitives: inner loop body and config helpers."""

import typing

import omegaconf
import torch
import transformers

from spe import generation, hooks


def run_single_sample(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    hook_fn: typing.Callable | None,
    hook_target: str,
    first_layer: int,
    last_layer: int,
    enable_thinking: bool = False,
) -> tuple[str, torch.Tensor, int, tuple[torch.Tensor, ...]]:
    """Register hooks, generate one token, remove hooks.

    Args:
        model: The language model.
        tokenizer: Matching tokenizer.
        messages: Chat messages for generation.
        hook_fn: Hook function to register, or ``None`` for
            baseline (no perturbation).
        hook_target: ``"attn"``, ``"mlp"``, or ``"both"``.
        first_layer: First layer to hook (inclusive).
        last_layer: Last layer to hook (inclusive, -1 = last).
        enable_thinking: Whether to allow thinking tokens.

    Returns:
        Tuple of (decoded_response, logits, generated_token_id, hidden_states).
    """
    if hook_fn is not None:
        handles = hooks.register_hooks(
            model,
            hook_fn,
            target=hook_target,
            first_layer=first_layer,
            last_layer=last_layer,
        )
    else:
        handles = []

    try:
        response, logits, token_id, hidden_states, _all_logits = (
            generation.generate_single_token(
                model,
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
            )
        )
    finally:
        hooks.remove_hooks(handles)

    return response, logits, token_id, hidden_states


def resolve_first_token(
    cfg: omegaconf.DictConfig,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
) -> int:
    """Return the first token index for the perturbation range.

    When ``perturb_from_turn`` is set, the index is computed
    dynamically from the tokenized messages. Otherwise the
    static ``first_token`` value is returned.
    """
    perturb_from_turn = cfg.perturbation.get("perturb_from_turn", None)
    if perturb_from_turn is not None:
        return generation.compute_turn_start_token(
            tokenizer,
            messages,
            perturb_from_turn,
            enable_thinking=cfg.model.thinking,
        )
    return cfg.perturbation.first_token


def resolve_last_token(
    cfg: omegaconf.DictConfig,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
) -> int:
    """Return the last token index for the perturbation range.

    When ``perturb_to_turn`` is set, the index is computed as
    the token before the start of that turn. Otherwise the
    static ``last_token`` value is returned.
    """
    perturb_to_turn = cfg.perturbation.get("perturb_to_turn", None)
    if perturb_to_turn is not None:
        if perturb_to_turn <= 0:
            raise ValueError(f"perturb_to_turn must be > 0 (got {perturb_to_turn}).")
        return (
            generation.compute_turn_start_token(
                tokenizer,
                messages,
                perturb_to_turn,
                enable_thinking=cfg.model.thinking,
            )
            - 1
        )
    return cfg.perturbation.last_token


def describe_option(name: str) -> str:
    """Return a parenthetical description for a class name."""
    descriptions = {
        "NOTHING": "(no perturbation)",
        "DROPOUT": "(random neuron dropping)",
        "NOISE": "(gaussian noise injection)",
    }
    return descriptions.get(name, "")
