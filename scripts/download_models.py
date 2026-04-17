"""Download all HF models used in sweeps."""

import os

os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-1b-it",
    "allenai/Olmo-3.1-32B-Instruct",
]

for model_id in MODELS:
    print(f"Downloading model {model_id}...")
    snapshot_download(model_id)
    print(f"Done: {model_id}\n")
