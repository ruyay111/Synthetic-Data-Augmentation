"""Utility API functions for diffusion pool loading and calibration.

Implementation lives in PoolProcessor. These snake_case functions re-export
pool I/O used by stitching and MVO sampling.
"""

from __future__ import annotations

from hmmdiff.data_collection.pool_processor import (
    affine_calibrate,
    check_pools,
    load_generated_images,
    load_generated_images_ew,
    load_pool,
    load_pool_meta,
    placeholder_pools,
    pool_demand,
    pool_dir,
    pools_available,
    regime_emission_stats,
    zscore_log_returns,
)

__all__ = [
    "affine_calibrate",
    "check_pools",
    "load_generated_images",
    "load_generated_images_ew",
    "load_pool",
    "load_pool_meta",
    "placeholder_pools",
    "pool_demand",
    "pool_dir",
    "pools_available",
    "regime_emission_stats",
    "zscore_log_returns",
]
