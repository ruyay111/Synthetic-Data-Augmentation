#!/usr/bin/env python3
"""Stage 3 (EW): train one UniTST_MP specialist per equal-weight volatility regime.

CLI around RegimeSpecialistDiffusion.TrainRegimes. Defaults to configs/ew.yaml.

Smoke test one regime before the full run:

  python scripts/03_train_specialists_ew.py --regimes 0 --epochs 2
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
    parser.add_argument("--config", default="configs/ew.yaml", help="Path to a YAML config.")
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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is None:
        args.epochs = int(os.environ.get("EPOCHS", cfg["diffusion"]["train_epochs"]))
    if args.loss is None:
        args.loss = os.environ.get("LOSS", cfg["diffusion"]["loss"])
    force = args.force or os.environ.get("FORCE", "0") == "1"
    trainer = RegimeSpecialistDiffusion(cfg)
    regimes = regime_list(args, int(cfg["regimes"]["n_regimes"]))
    print(f"[INFO] python={sys.executable} device={args.device} epochs={args.epochs}")
    print(f"[INFO] regimes={regimes} loss={args.loss}")
    print(f"[INFO] windows={config_path(cfg, 'regime_windows')}")
    trainer.TrainRegimes(
        cfg,
        regimes,
        device=args.device,
        epochs=args.epochs,
        loss=args.loss,
        force=force,
        skip_eval=args.skip_eval,
        eval_only=args.eval_only,
    )
    print("[OK] done. Next: python scripts/04_generate_pools_ew.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
