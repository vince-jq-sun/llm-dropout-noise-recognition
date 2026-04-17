"""Shared wandb sweep data loading with local caching."""

import pathlib

import pandas as pd
import wandb
from tqdm.auto import tqdm

CACHE_DIR = pathlib.Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_KEY = "accuracy_primary"

_api = None


def _get_api() -> wandb.Api:
    global _api
    if _api is None:
        _api = wandb.Api()
    return _api


def load_sweep(
    sweep_id: str, project: str, required_key: str = REQUIRED_KEY
) -> pd.DataFrame:
    """Download all runs from a sweep, caching to a local pickle file.

    Args:
        sweep_id: The wandb sweep identifier.
        project: The wandb project name (e.g. "llm-mechanistic-detection").
        required_key: Only include runs whose summary contains this key.

    Returns:
        DataFrame with one row per run (config + summary merged).
    """
    cache_path = CACHE_DIR / f"{sweep_id}.pkl"
    if cache_path.exists():
        print(f"Loading {sweep_id} from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    api = _get_api()
    print(f"Downloading {sweep_id} from wandb...")
    runs_iter = api.runs(project, filters={"sweep": sweep_id}, per_page=1000)

    rows = []
    skipped = 0
    for run in tqdm(runs_iter, desc="Fetching runs"):
        summary = run.summary._json_dict
        if required_key not in summary:
            skipped += 1
            continue
        row = {**run.config, **summary}
        row["_run_state"] = run.state
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle(cache_path)
    print(
        f"  Total runs: {len(rows) + skipped}, with '{required_key}': {len(rows)}, missing: {skipped}"
    )
    return df


def load_sweep_tables(
    sweep_id: str,
    project: str,
    table_key: str = "results_table_full",
    required_key: str = REQUIRED_KEY,
) -> pd.DataFrame:
    """Download per-sample results tables for all runs in a sweep.

    Each run's wandb Table is fetched and concatenated into a single DataFrame.
    Run-level config columns are merged so each row has both per-sample data
    and the run's config (model, perturbation params, etc.).

    Args:
        sweep_id: The wandb sweep identifier.
        project: The wandb project name.
        table_key: The wandb log key for the table (default "results_table_full").
        required_key: Only include runs whose summary contains this key.

    Returns:
        DataFrame with one row per sample across all runs.
    """
    cache_path = CACHE_DIR / f"{sweep_id}_tables.pkl"
    if cache_path.exists():
        print(f"Loading {sweep_id} tables from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    api = _get_api()
    print(f"Downloading {sweep_id} per-sample tables from wandb...")
    all_runs = list(api.runs(project, filters={"sweep": sweep_id}, per_page=1000))

    all_rows = []
    skipped = 0
    for run in tqdm(all_runs, desc="Fetching tables"):
        summary = run.summary._json_dict
        if required_key not in summary:
            skipped += 1
            continue

        # Extract run-level config for merging
        run_config = {
            "_run_id": run.id,
            "_run_name": run.name,
        }
        for k, v in run.config.items():
            if not isinstance(v, dict):
                run_config[k] = v

        # Fetch the table artifact
        table_ref = summary.get(table_key)
        if table_ref is None:
            skipped += 1
            continue

        try:
            # Find the logged artifact that contains the table
            artifact = None
            for art in run.logged_artifacts():
                if table_key in [f.name.split(".table.json")[0] for f in art.files()]:
                    artifact = art
                    break
            if artifact is None:
                skipped += 1
                continue
            table_file = artifact.get(table_key)
            table_df = pd.DataFrame(table_file.data, columns=table_file.columns)
            for k, v in run_config.items():
                table_df[k] = v
            all_rows.append(table_df)
        except Exception as e:
            print(f"  Warning: failed to fetch table for run {run.id}: {e}")
            skipped += 1
            continue

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df.to_pickle(cache_path)
    print(
        f"  Total runs: {len(all_runs)}, fetched tables: {len(all_rows)}, skipped: {skipped}"
    )
    print(f"  Total samples: {len(df)}")
    return df


def clear_cache(sweep_id: str) -> None:
    """Delete cached files for a sweep, forcing re-download."""
    for name in [f"{sweep_id}.pkl", f"{sweep_id}_tables.pkl"]:
        cache_path = CACHE_DIR / name
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_path}")
