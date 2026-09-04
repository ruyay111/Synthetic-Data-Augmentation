#!/usr/bin/env python3
"""Stage 4 (EW): sample specialist pools for equal-weight regimes.

CLI around RegimeSpecialistDiffusion.GeneratePool. Writes
data/pools/ew/regime_k{k}/windows.npy in raw log-return units.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import config_path, load_config  # noqa: E402
from hmmdiff.model_design.diffusion_trainer import RegimeSpecialistDiffusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ew.yaml")
    parser.add_argument("--n-pool", type=int, default=None, help="Override config sampling.n_pool.")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, mps, or cpu. Default: checkpoint device with auto-fallback if unavailable.",
    )
    parser.add_argument("--regimes", type=int, nargs="*", default=None)
    parser.add_argument("--description-template", default="specialist_regime_{k}")
    parser.add_argument("--force", action="store_true", help="Resample regimes that already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = RegimeSpecialistDiffusion(cfg)
    checkpoints_root = config_path(cfg, "checkpoints")
    pools_root = config_path(cfg, "pools")
    pools_root.mkdir(parents=True, exist_ok=True)

    n_pool = args.n_pool or int(cfg["sampling"]["n_pool"])
    regimes = args.regimes if args.regimes is not None else range(int(cfg["regimes"]["n_regimes"]))

    cwd = os.getcwd()
    try:
        for regime in regimes:
            out_dir = pools_root / f"regime_k{regime}"
            if (out_dir / "windows.npy").exists() and not args.force:
                print(f"[SKIP] regime {regime}: pool exists (--force to resample)")
                continue
            description = args.description_template.format(k=regime)
            try:
                checkpoint_dir = trainer.FindCheckpointDir(checkpoints_root, description)
            except FileNotFoundError as exc:
                print(f"[SKIP] regime {regime}: {exc}")
                continue
            print(f"[GEN] regime {regime} from {checkpoint_dir.name} (n_pool={n_pool})")
            windows = trainer.GeneratePool(checkpoint_dir, cfg, n_pool, args.device, regime)
            meta = trainer.WritePool(windows, out_dir, regime, cfg, checkpoint_dir)
            print(
                f"[OK] {out_dir / 'windows.npy'} shape={windows.shape} "
                f"mean={meta['mean']:.4f} var={meta['var']:.4f}"
            )
    finally:
        os.chdir(cwd)

    print("[OK] done. Validate with HMM-Diffusion-EW.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
