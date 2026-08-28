#!/usr/bin/env python3
"""Stage 3: train one unconditional UniTST_MP diffusion specialist per volatility regime.

Windows-friendly counterpart of ``03_train_specialists.sh``. Calls ``reference_model/diffusion/run.py``
with the same flags. Absolute paths are required because ``run.py`` chdirs to its own directory.

After each specialist trains, ``test()`` writes dist/autocorr/moments/cov/corr plots at
``sampling.sample_step`` (450) only, under ``test_results/<run>/450_discrete_DDPM_<temp>/``.

Smoke test one regime before the full run:

  python scripts/03_train_specialists.py --regimes 0 --epochs 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import config_path, load_config  # noqa: E402

MIN_WINDOWS = 8
RUN_PY = Path(__file__).resolve().parents[1] / "reference_model" / "diffusion" / "run.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        help="cuda, mps, or cpu. Default: DEVICE env or cuda.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--regimes",
        type=int,
        nargs="*",
        default=None,
        help="Regimes to train. Default: REGIMES env or all.",
    )
    parser.add_argument("--loss", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain regimes that already have a checkpoint. Also honors FORCE=1.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only; do not write dist/autocorr plots.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; run the sample_step=450 plots on existing checkpoints.",
    )
    return parser.parse_args()


def regime_list(args: argparse.Namespace, n_regimes: int) -> list[int]:
    if args.regimes:
        return list(args.regimes)
    env = os.environ.get("REGIMES", "").strip()
    if env:
        return [int(part) for part in env.replace(",", " ").split()]
    return list(range(n_regimes))


def has_checkpoint(checkpoints_dir: Path, regime: int) -> bool:
    return any(checkpoints_dir.glob(f"*_specialist_regime_{regime}/checkpoint.pth"))


def run_one(
    regime: int,
    data_path: Path,
    args: argparse.Namespace,
    cfg: dict,
    *,
    skip_train: bool,
    skip_eval: bool,
) -> None:
    diffusion = cfg["diffusion"]
    sampling = cfg["sampling"]
    cmd = [
        sys.executable,
        str(RUN_PY),
        "--task_name",
        "diffusion_denoised_x",
        "--model",
        "UniTST_MP",
        "--data",
        "RegimeWindows",
        "--data_path",
        str(data_path),
        "--checkpoints",
        str(config_path(cfg, "checkpoints")) + os.sep,
        "--test_results",
        str(config_path(cfg, "test_results")) + os.sep,
        "--seq_len",
        str(diffusion["seq_len"]),
        "--enc_in",
        str(diffusion["enc_in"]),
        "--scale",
        str(diffusion["scale"]),
        "--batch_size",
        str(diffusion["batch_size"]),
        "--sample_multiplier",
        str(diffusion["sample_multiplier"]),
        "--train_epochs",
        str(args.epochs),
        "--causal_mask",
        "--ind_proj",
        "--RoPE",
        "--channel_embed",
        "--learning_rate",
        str(diffusion["learning_rate"]),
        "--lr_decay_rounds",
        str(diffusion["lr_decay_rounds"]),
        "--loss",
        args.loss,
        "--device",
        args.device,
        "--gpu",
        "0",
        "--description",
        f"specialist_regime_{regime}",
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_eval:
        cmd.append("--skip_test")
    else:
        cmd.extend(
            [
                "--eval_sample_step",
                str(sampling["sample_step"]),
                "--eval_temperature",
                str(sampling["temperature"]),
            ]
        )
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is None:
        args.epochs = int(os.environ.get("EPOCHS", cfg["diffusion"]["train_epochs"]))
    if args.loss is None:
        args.loss = os.environ.get("LOSS", cfg["diffusion"]["loss"])
    force = args.force or os.environ.get("FORCE", "0") == "1"
    skip_eval = args.skip_eval
    eval_only = args.eval_only
    sample_step = int(cfg["sampling"]["sample_step"])

    windows_dir = config_path(cfg, "regime_windows")
    checkpoints_dir = config_path(cfg, "checkpoints")
    test_results_dir = config_path(cfg, "test_results")
    test_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    regimes = regime_list(args, int(cfg["regimes"]["n_regimes"]))
    print(f"[INFO] python={sys.executable} device={args.device} epochs={args.epochs}")
    print(f"[INFO] regimes={regimes} loss={args.loss}")
    print(f"[INFO] windows={windows_dir}")
    if skip_eval:
        print("[INFO] eval skipped")
    else:
        print(f"[INFO] eval sample_step={sample_step} -> {test_results_dir}")

    for regime in regimes:
        data_path = windows_dir / f"regime_{regime}.npy"
        if not data_path.exists():
            print(f"[SKIP] regime {regime}: {data_path} missing; run scripts/02_build_diffusion_dataset.py")
            continue
        n_windows = int(np.load(data_path).shape[0])
        if n_windows < MIN_WINDOWS:
            print(f"[SKIP] regime {regime}: only {n_windows} windows")
            continue
        exists = has_checkpoint(checkpoints_dir, regime)
        if eval_only:
            if not exists:
                print(f"[SKIP] regime {regime}: no checkpoint; train first")
                continue
            print(f"[EVAL] regime {regime}: sample_step={sample_step}")
            run_one(regime, data_path, args, cfg, skip_train=True, skip_eval=False)
            continue
        if exists and not force:
            if skip_eval:
                print(f"[SKIP] regime {regime}: checkpoint exists (--force to retrain)")
                continue
            print(
                f"[EVAL] regime {regime}: checkpoint exists, plotting sample_step={sample_step} "
                "(--force to retrain)"
            )
            run_one(regime, data_path, args, cfg, skip_train=True, skip_eval=False)
            continue
        print(f"[TRAIN] regime {regime}: {n_windows} windows")
        run_one(regime, data_path, args, cfg, skip_train=False, skip_eval=skip_eval)

    print("[OK] done. Next: python scripts/04_generate_pools.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
