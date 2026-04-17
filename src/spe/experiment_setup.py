"""Shared experiment setup and teardown used by all entry points."""

import random
import typing

import numpy as np
import omegaconf
import torch
import transformers

from spe import model_utils, wandb_utils


def setup_experiment(
    cfg: omegaconf.DictConfig,
) -> tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Print config, set seeds, init wandb and load model.

    Args:
        cfg: Resolved Hydra config.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(omegaconf.OmegaConf.to_yaml(cfg))

    seed = cfg.experiment.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    wandb_utils.init_run(cfg)

    model_utils.print_hardware_info()

    model, tokenizer = model_utils.load_model_and_tokenizer(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
    )

    return model, tokenizer


def teardown_experiment(
    metrics: dict[str, typing.Any],
    results: list[dict[str, typing.Any]],
    mode: str,
    cfg: omegaconf.DictConfig,
) -> None:
    """Log results to wandb and finish the run.

    Args:
        metrics: Metrics dict returned by the evaluation module.
        results: List of result dicts (one for each sample).
        mode: Experiment mode string for wandb grouping.
        cfg: Resolved Hydra config.
    """
    wandb_utils.log_results(metrics, results, mode=mode, cfg=cfg)
    wandb_utils.finish_run()

    print(f"\nExperiment complete. Mode: {mode}")
