"""Utility API functions for per-regime diffusion windows.

Implementation lives in WindowProcessor. These snake_case functions re-export
window cutting, tiling, and manifest I/O.
"""

from __future__ import annotations

from hmmdiff.data_collection.window_processor import (
    build_windows,
    contiguous_runs,
    load_manifest,
    save_windows,
    sliding_windows,
    tiled_windows,
)

__all__ = [
    "build_windows",
    "contiguous_runs",
    "load_manifest",
    "save_windows",
    "sliding_windows",
    "tiled_windows",
]
