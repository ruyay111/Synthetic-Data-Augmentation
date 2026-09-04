"""Diffusion training and sampling.

UnconditionalDiffusion is the regime-free DDPM. RegimeSpecialistDiffusion
trains one UniTST_MP specialist per volatility regime and samples pools.
Training launches reference_model/diffusion/run.py; sampling loads
checkpoints through Exp_Diffusion_Denoised_X.
Literature: Ho, Jain, and Abbeel (2020).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hmmdiff.config import REPO_ROOT, config_path
from hmmdiff.constants import (
    DDIM_DISCRETIZE,
    DDIM_ETA,
    DDIM_N_STEPS,
    DIFFUSION_DATA_NAME,
    DIFFUSION_MODEL_NAME,
    DIFFUSION_TASK_NAME,
    MIN_WINDOWS,
    OVERLAP_RATIO,
)
from hmmdiff.model_design.base_model import BaseDiffusionModel

RUN_PY = REPO_ROOT / "reference_model" / "diffusion" / "run.py"
DIFFUSION_DIR = REPO_ROOT / "reference_model" / "diffusion"


class UnconditionalDiffusion(BaseDiffusionModel):
    """Single DDPM trained on all windows, with no regime conditioning."""

    name = "Unconditional Diffusion"

    def Fit(self, train_data: Any, cfg: dict[str, Any] | None = None) -> Any:
        """
        Document the unconditional generator. Training is performed offline
        and written to trained_diffusion_withoutregime/generated_data.csv.

        Parameters:
        train_data: any
            Unused; the unconditional run lives in the reference trainer.
        cfg: dict or None
            Unused.

        Return:
           None
        """
        del train_data, cfg
        return None

    def LoadWindows(self, path: Path, seq_len: int, n_assets: int) -> np.ndarray:
        """
        Reshape generated_data.csv to (n_windows, seq_len, n_assets).

        Parameters:
        path: pathlib.Path
            CSV of flattened log-return windows.
        seq_len: int
            Window length.
        n_assets: int
            Number of assets.

        Return:
           numpy.ndarray of log-return windows.
        """
        from hmmdiff.mvo.mixed_sample import load_uncond_windows

        return load_uncond_windows(path, seq_len=seq_len, n_assets=n_assets)


class RegimeSpecialistDiffusion(BaseDiffusionModel):
    """One diffusion specialist per volatility regime, plus pool sampling."""

    name = "HMM-Diffusion Specialist"

    def __init__(self, cfg: dict[str, Any] | None = None):
        """
        Store an optional pipeline config.

        Parameters:
        cfg: dict or None
            YAML mapping with diffusion, sampling, and path keys.

        Return:
           None
        """
        self.cfg = cfg

    def Fit(self, train_data: Any, cfg: dict[str, Any] | None = None) -> Any:
        """
        Train specialists for every regime that has enough windows.

        Parameters:
        train_data: any
            Unused; windows are read from cfg paths.regime_windows.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           None
        """
        del train_data
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("Fit requires a config dict.")
        n_regimes = int(cfg["regimes"]["n_regimes"])
        self.TrainRegimes(cfg, list(range(n_regimes)))
        return None

    def HasCheckpoint(self, checkpoints_dir: Path, regime: int) -> bool:
        """
        Whether a specialist checkpoint exists for a regime.

        Parameters:
        checkpoints_dir: pathlib.Path
            Checkpoint root.
        regime: int
            Regime index.

        Return:
           bool
        """
        return any(checkpoints_dir.glob(f"*_specialist_regime_{regime}/checkpoint.pth"))

    def TrainRegimes(
        self,
        cfg: dict[str, Any],
        regimes: list[int],
        *,
        device: str = "cuda",
        epochs: int | None = None,
        loss: str | None = None,
        force: bool = False,
        skip_eval: bool = False,
        eval_only: bool = False,
    ) -> None:
        """
        Train or evaluate UniTST_MP specialists.

        Parameters:
        cfg: dict
            Pipeline config.
        regimes: list
            Regime indices to train.
        device: str
            cuda, mps, or cpu.
        epochs: int or None
            Override train_epochs.
        loss: str or None
            Override diffusion loss string.
        force: bool
            Retrain even if a checkpoint exists.
        skip_eval: bool
            Skip dist/autocorr plots.
        eval_only: bool
            Skip training and only plot existing checkpoints.

        Return:
           None
        """
        diffusion = cfg["diffusion"]
        epochs = int(epochs if epochs is not None else diffusion["train_epochs"])
        loss = str(loss if loss is not None else diffusion["loss"])
        windows_dir = config_path(cfg, "regime_windows")
        checkpoints_dir = config_path(cfg, "checkpoints")
        test_results_dir = config_path(cfg, "test_results")
        test_results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        for regime in regimes:
            data_path = windows_dir / f"regime_{regime}.npy"
            if not data_path.exists():
                print(f"[SKIP] regime {regime}: {data_path} missing")
                continue
            n_windows = int(np.load(data_path).shape[0])
            if n_windows < MIN_WINDOWS:
                print(f"[SKIP] regime {regime}: only {n_windows} windows")
                continue
            exists = self.HasCheckpoint(checkpoints_dir, regime)
            if eval_only:
                if not exists:
                    print(f"[SKIP] regime {regime}: no checkpoint; train first")
                    continue
                self.RunOne(
                    regime,
                    data_path,
                    cfg,
                    device=device,
                    epochs=epochs,
                    loss=loss,
                    skip_train=True,
                    skip_eval=False,
                )
                continue
            if exists and not force:
                if skip_eval:
                    print(f"[SKIP] regime {regime}: checkpoint exists")
                    continue
                self.RunOne(
                    regime,
                    data_path,
                    cfg,
                    device=device,
                    epochs=epochs,
                    loss=loss,
                    skip_train=True,
                    skip_eval=False,
                )
                continue
            print(f"[TRAIN] regime {regime}: {n_windows} windows")
            self.RunOne(
                regime,
                data_path,
                cfg,
                device=device,
                epochs=epochs,
                loss=loss,
                skip_train=False,
                skip_eval=skip_eval,
            )

    def RunOne(
        self,
        regime: int,
        data_path: Path,
        cfg: dict[str, Any],
        *,
        device: str,
        epochs: int,
        loss: str,
        skip_train: bool,
        skip_eval: bool,
    ) -> None:
        """
        Launch run.py for one specialist.

        Parameters:
        regime: int
            Regime index.
        data_path: pathlib.Path
            Path to regime_{k}.npy.
        cfg: dict
            Pipeline config.
        device: str
            Training device.
        epochs: int
            Number of epochs.
        loss: str
            Loss string.
        skip_train: bool
            Pass --skip_train.
        skip_eval: bool
            Pass --skip_test.

        Return:
           None
        """
        diffusion = cfg["diffusion"]
        sampling = cfg["sampling"]
        cmd = [
            sys.executable,
            str(RUN_PY),
            "--task_name",
            DIFFUSION_TASK_NAME,
            "--model",
            DIFFUSION_MODEL_NAME,
            "--data",
            DIFFUSION_DATA_NAME,
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
            str(epochs),
            "--causal_mask",
            "--ind_proj",
            "--RoPE",
            "--channel_embed",
            "--learning_rate",
            str(diffusion["learning_rate"]),
            "--lr_decay_rounds",
            str(diffusion["lr_decay_rounds"]),
            "--loss",
            loss,
            "--device",
            device,
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

    def FindCheckpointDir(self, checkpoints_root: Path, description: str) -> Path:
        """
        Newest checkpoint directory whose name ends with _{description}.

        Parameters:
        checkpoints_root: pathlib.Path
            Checkpoint root.
        description: str
            Run description, e.g. specialist_regime_0.

        Return:
           pathlib.Path of the checkpoint directory.
        """
        matches = [
            path
            for path in checkpoints_root.glob("*")
            if path.is_dir() and path.name.endswith(f"_{description}")
        ]
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint directory ending with _{description} under {checkpoints_root}."
            )
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def GeneratePool(
        self,
        checkpoint_dir: Path,
        cfg: dict[str, Any],
        n_pool: int,
        device: str | None,
        regime: int,
    ) -> np.ndarray:
        """
        Sample n_pool windows from one specialist checkpoint.

        Parameters:
        checkpoint_dir: pathlib.Path
            Directory containing checkpoint.pth and args.json.
        cfg: dict
            Pipeline config.
        n_pool: int
            Number of windows to sample.
        device: str or None
            Requested device; None uses the checkpoint device with fallback.
        regime: int
            Regime index (used to locate windows for the scaler).

        Return:
           numpy.ndarray of shape (n_pool, seq_len, n_channels).
        """
        os.chdir(DIFFUSION_DIR)
        if str(DIFFUSION_DIR) not in sys.path:
            sys.path.insert(0, str(DIFFUSION_DIR))
        from src.exp.exp_diffusion_denoised_x import Exp_Diffusion_Denoised_X

        args = self.LoadRunArgs(checkpoint_dir, device, regime=regime, cfg=cfg)
        seq_len = int(args.seq_len)
        n_channels = int(args.enc_in)
        exp = Exp_Diffusion_Denoised_X(args)
        checkpoint = checkpoint_dir / "checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing {checkpoint}")
        exp.model.load_state_dict(self.LoadCheckpointState(checkpoint))
        dataset, _ = exp._get_data()
        sampling = cfg["sampling"]
        generated = exp.generate_data(
            size=n_pool,
            sample_step=int(sampling["sample_step"]),
            dataset=dataset,
            model=exp.model,
            sampler=sampling["sampler"],
            n_steps=DDIM_N_STEPS,
            ddim_discretize=DDIM_DISCRETIZE,
            ddim_eta=DDIM_ETA,
            method=sampling["method"],
            overlap_ratio=OVERLAP_RATIO,
            temperature=float(sampling["temperature"]),
        )
        flat = generated.to_numpy(dtype=float)
        expected = (n_pool * seq_len, n_channels)
        if flat.shape != expected:
            raise ValueError(f"Generated shape {flat.shape} != {expected}")
        return flat.reshape(n_pool, seq_len, n_channels)

    def LoadRunArgs(
        self, checkpoint_dir: Path, device: str | None, *, regime: int, cfg: dict[str, Any]
    ) -> Namespace:
        """
        Load args.json and point data_path at the local windows file.

        Parameters:
        checkpoint_dir: pathlib.Path
            Checkpoint directory.
        device: str or None
            Requested device.
        regime: int
            Regime index.
        cfg: dict
            Pipeline config.

        Return:
           argparse.Namespace for Exp_Diffusion_Denoised_X.
        """
        args_path = checkpoint_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"Missing {args_path}")
        args = Namespace(**json.loads(args_path.read_text(encoding="utf-8")))
        if not hasattr(args, "compile"):
            args.compile = False
        resolved = self.ResolvePoolDevice(
            device, getattr(args, "device", "cuda"), getattr(args, "use_gpu", True)
        )
        args.device = resolved
        args.use_gpu = resolved != "cpu"
        windows_path = config_path(cfg, "regime_windows") / f"regime_{regime}.npy"
        if not windows_path.exists():
            raise FileNotFoundError(f"{windows_path} not found.")
        args.data_path = str(windows_path)
        return args

    def ResolvePoolDevice(
        self, requested: str | None, checkpoint_device: str, use_gpu: bool
    ) -> str:
        """
        Use the checkpoint device when available; otherwise fall back to MPS or CPU.

        Parameters:
        requested: str or None
            User override.
        checkpoint_device: str
            Device stored in args.json.
        use_gpu: bool
            Whether the run was configured with a GPU.

        Return:
           str device name.
        """
        import torch

        if requested is not None:
            return requested
        device = str(checkpoint_device or "auto")
        if device == "auto":
            return "auto"
        if not use_gpu or device == "cpu":
            return "cpu"
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("[WARN] Checkpoint was trained on CUDA; sampling on MPS instead.")
                return "mps"
            print("[WARN] Checkpoint was trained on CUDA; sampling on CPU instead.")
            return "cpu"
        if device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            print("[WARN] Checkpoint requested MPS; sampling on CPU instead.")
            return "cpu"
        return device

    def LoadCheckpointState(self, checkpoint: Path) -> dict:
        """
        Load a specialist checkpoint on CPU so CUDA-trained weights work on Mac.

        Parameters:
        checkpoint: pathlib.Path
            Path to checkpoint.pth.

        Return:
           dict of state_dict tensors.
        """
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return {key.replace("_orig_mod.", ""): value for key, value in state.items()}

    def WritePool(
        self,
        windows: np.ndarray,
        out_dir: Path,
        regime: int,
        cfg: dict[str, Any],
        checkpoint_dir: Path,
    ) -> dict[str, Any]:
        """
        Save windows.npy and meta.json for one regime pool.

        Parameters:
        windows: numpy.ndarray
            Sampled windows.
        out_dir: pathlib.Path
            Output directory.
        regime: int
            Regime index.
        cfg: dict
            Pipeline config.
        checkpoint_dir: pathlib.Path
            Source checkpoint.

        Return:
           dict metadata.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "windows.npy", windows)
        try:
            rel = str(checkpoint_dir.resolve().relative_to(REPO_ROOT))
        except ValueError:
            rel = str(checkpoint_dir)
        meta = {
            "regime": int(regime),
            "n_pool": int(windows.shape[0]),
            "seq_len": int(windows.shape[1]),
            "n_channels": int(windows.shape[2]),
            "sample_step": int(cfg["sampling"]["sample_step"]),
            "temperature": float(cfg["sampling"]["temperature"]),
            "sampler": cfg["sampling"]["sampler"],
            "method": cfg["sampling"]["method"],
            "checkpoint_dir": rel,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "units": "raw_log_returns",
            "mean": float(windows.mean()),
            "var": float(windows.var()),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
