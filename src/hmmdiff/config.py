"""Configuration loading and import bootstrapping."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Defaults to ``configs/default.yaml``."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    """Turn a config-relative path into an absolute one."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def config_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve one entry of the config's ``paths`` block."""
    try:
        raw = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"No path named {key!r} in config['paths']") from exc
    return resolve(raw)


def bootstrap_imports(diffusion: bool = False) -> None:
    """Put the reference model packages and ``src`` on ``sys.path``.

    ``reference_model/hmmgan`` is always needed. The diffusion tree is opt-in because importing it pulls
    in torch, which is slow and unnecessary for the HMM-only parts of the pipeline. Note that the
    diffusion tree has no ``__init__.py`` files and expects its own root on ``sys.path`` so that
    ``import src.exp...`` resolves; see ``reference_model/VENDORED.md``.
    """
    entries = [REPO_ROOT / "src", REPO_ROOT / "reference_model"]
    if diffusion:
        entries.append(REPO_ROOT / "reference_model" / "diffusion")
    for entry in entries:
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
