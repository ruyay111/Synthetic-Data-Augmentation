"""HMM-Diffusion: per-regime diffusion generators feeding the supervised HMM pipeline.

Reproduces ``Supervised HMMs.ipynb`` with the per-regime GAN generators replaced by per-regime
diffusion specialists. See ``PLAN.md``.
"""

from .config import bootstrap_imports, config_path, load_config, resolve, REPO_ROOT

__all__ = ["bootstrap_imports", "config_path", "load_config", "resolve", "REPO_ROOT"]
