"""Model and tokenizer loading, hardware diagnostics."""

import typing

import torch
import transformers


def print_hardware_info() -> None:
    """Print GPU / CUDA availability and memory usage."""
    print("=" * 50)
    print("HARDWARE CHECK")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, will run on CPU (slow!)")
    print("=" * 50)


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. ``Qwen/Qwen3-4B-Instruct-2507``).
        dtype: Torch dtype string.  Converted via ``getattr(torch, dtype)``.
        device_map: Device map strategy passed to ``from_pretrained``.

    Returns:
        Tuple of (model, tokenizer).
    """
    torch_dtype = getattr(torch, dtype)

    print(f"Loading model: {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    print(f"Model loaded on device: {model.device}")  # type: ignore[attr-defined]
    print(f"Model dtype: {model.dtype}")  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        print(
            f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    return typing.cast(transformers.AutoModelForCausalLM, model), typing.cast(
        transformers.AutoTokenizer, tokenizer
    )
