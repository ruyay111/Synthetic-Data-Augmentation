#!/usr/bin/env python3
"""Stage 4: sample a pool of synthetic windows from each trained specialist.

Univariate adaptation of ``generate_specialist_pools.py`` from the ruya tree. Writes

  data/pools/regime_k{k}/windows.npy   (n_pool, seq_len, 1) in z-scored log-return units
  data/pools/regime_k{k}/meta.json

The output is already in the notebook's ``emission`` units: the dataset fits a QuantileTransformer on
the training windows and ``generate_data`` inverse-transforms before returning, so training on
z-scored input yields z-scored output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hmmdiff.config import config_path, load_config  # noqa: E402

DIFFUSION_DIR = REPO_ROOT / "third_party" / "diffusion"
# run.py chdirs to the diffusion directory; the experiment classes assume that layout, so match it.
os.chdir(DIFFUSION_DIR)
sys.path.insert(0, str(DIFFUSION_DIR))

from src.exp.exp_diffusion_denoised_x import Exp_Diffusion_Denoised_X  # noqa: E402
from src.utils.utils import process_model_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-pool", type=int, default=None, help="Override config sampling.n_pool.")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu. Default: checkpoint's.")
    parser.add_argument("--regimes", type=int, nargs="*", default=None)
    parser.add_argument("--description-template", default="specialist_regime_{k}")
    parser.add_argument("--force", action="store_true", help="Resample regimes that already exist.")
    return parser.parse_args()


def find_checkpoint_dir(checkpoints_root: Path, description: str) -> Path:
    matches = [
        path
        for path in checkpoints_root.glob("*")
        if path.is_dir() and path.name.endswith(f"_{description}")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint directory ending with _{description} under {checkpoints_root}. "
            "Run scripts/03_train_specialists.sh first."
        )
    # Several runs can share a description if training was repeated; the newest one wins.
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_run_args(checkpoint_dir: Path, device: str | None) -> Namespace:
    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing {args_path}")
    args = Namespace(**json.loads(args_path.read_text(encoding="utf-8")))
    if not hasattr(args, "compile"):
        args.compile = False
    if device is not None:
        args.device = device
        args.use_gpu = device != "cpu"
    return args


def generate_pool(checkpoint_dir: Path, cfg: dict, n_pool: int, device: str | None) -> np.ndarray:
    args = load_run_args(checkpoint_dir, device)
    seq_len = int(args.seq_len)
    n_channels = int(args.enc_in)

    exp = Exp_Diffusion_Denoised_X(args)
    checkpoint = checkpoint_dir / "checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}")
    exp.model.load_state_dict(process_model_dict(str(checkpoint)))

    # The dataset is reloaded because generate_data needs its fitted scaler to invert the transform.
    dataset, _ = exp._get_data()
    sampling = cfg["sampling"]
    generated = exp.generate_data(
        size=n_pool,
        sample_step=int(sampling["sample_step"]),
        dataset=dataset,
        model=exp.model,
        sampler=sampling["sampler"],
        n_steps=20,
        ddim_discretize="uniform",
        ddim_eta=0.0,
        method=sampling["method"],
        overlap_ratio=0.25,
        temperature=float(sampling["temperature"]),
    )

    flat = generated.to_numpy(dtype=float)
    expected = (n_pool * seq_len, n_channels)
    if flat.shape != expected:
        raise ValueError(f"Generated shape {flat.shape} != {expected}")
    return flat.reshape(n_pool, seq_len, n_channels)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoints_root = config_path(cfg, "checkpoints")
    pools_root = config_path(cfg, "pools")
    pools_root.mkdir(parents=True, exist_ok=True)

    n_pool = args.n_pool or int(cfg["sampling"]["n_pool"])
    regimes = args.regimes if args.regimes is not None else range(int(cfg["regimes"]["n_regimes"]))

    for regime in regimes:
        out_dir = pools_root / f"regime_k{regime}"
        if (out_dir / "windows.npy").exists() and not args.force:
            print(f"[SKIP] regime {regime}: pool exists (--force to resample)")
            continue

        description = args.description_template.format(k=regime)
        try:
            checkpoint_dir = find_checkpoint_dir(checkpoints_root, description)
        except FileNotFoundError as exc:
            print(f"[SKIP] regime {regime}: {exc}")
            continue

        print(f"[GEN] regime {regime} from {checkpoint_dir.name} (n_pool={n_pool})")
        windows = generate_pool(checkpoint_dir, cfg, n_pool, args.device)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "windows.npy", windows)
        meta = {
            "regime": int(regime),
            "n_pool": int(windows.shape[0]),
            "seq_len": int(windows.shape[1]),
            "n_channels": int(windows.shape[2]),
            "sample_step": int(cfg["sampling"]["sample_step"]),
            "temperature": float(cfg["sampling"]["temperature"]),
            "sampler": cfg["sampling"]["sampler"],
            "method": cfg["sampling"]["method"],
            "checkpoint_dir": str(checkpoint_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "units": "z_scored_log_returns",
            "mean": float(windows.mean()),
            "var": float(windows.var()),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[OK] {out_dir / 'windows.npy'} shape={windows.shape} "
            f"mean={meta['mean']:.4f} var={meta['var']:.4f}"
        )

    print("[OK] done. Validate with the notebook's pool checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
