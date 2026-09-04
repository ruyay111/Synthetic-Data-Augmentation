"""Configuration loading and import bootstrapping.

load_config, resolve, and config_path read YAML. Numeric defaults that are
not file paths live in hmmdiff.constants.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read the YAML config.

    Parameters:
    path: str, pathlib.Path, or None
        Config file. Defaults to configs/default.yaml.

    Return:
       dict loaded from YAML.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    """
    Turn a config-relative path into an absolute one.

    Parameters:
    path: str or pathlib.Path
        Path relative to the repo root, or already absolute.

    Return:
       pathlib.Path.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def config_path(cfg: dict[str, Any], key: str) -> Path:
    """
    Resolve one entry of the config's paths block.

    Parameters:
    cfg: dict
        Loaded YAML mapping.
    key: str
        Name under cfg['paths'].

    Return:
       pathlib.Path.
    """
    try:
        raw = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"No path named {key!r} in config['paths']") from exc
    return resolve(raw)


def bootstrap_imports(diffusion: bool = False) -> None:
    """
    Put the reference model packages and src on sys.path.

    Parameters:
    diffusion: bool
        If true, also add reference_model/diffusion so import src.exp works.

    Return:
       None
    """
    entries = [REPO_ROOT / "src", REPO_ROOT / "reference_model"]
    if diffusion:
        entries.append(REPO_ROOT / "reference_model" / "diffusion")
    for entry in entries:
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
