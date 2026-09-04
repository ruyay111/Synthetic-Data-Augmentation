"""Model-design base classes.

BaseGenerativeModel is the top-level abstraction for models that consume a
processed return panel. BaseHiddenMarkovModel adds train/test helpers shared
by the HMM variants. BaseDiffusionModel is the parent of unconditional and
regime-specialist diffusion trainers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseGenerativeModel(ABC):
    """Shared interface for models used in synthetic data augmentation.

    Subclasses implement Fit on processed training data. Optional hooks cover
    train/test splitting that already happened in PriceReturnProcessor.
    """

    name = "BaseGenerativeModel"

    def SplitTrainTest(
        self, returns: pd.DataFrame, column: str = "z_return"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slice one column into train and test using the split column.

        Parameters:
        returns: pandas.DataFrame
            Frame with split and the requested column.
        column: str
            Column to slice.

        Return:
           tuple of (train array, test array).
        """
        mask = returns["split"].to_numpy() == "train"
        values = returns[column].to_numpy(dtype=float)
        return values[mask], values[~mask]

    @abstractmethod
    def Fit(self, train_data: Any, cfg: dict[str, Any] | None = None) -> Any:
        """
        Fit the model on training data.

        Parameters:
        train_data: any
            Prepared training object (HMM frame or window array).
        cfg: dict or None
            Pipeline config.

        Return:
           Fitted model object.
        """
        raise NotImplementedError


class BaseHiddenMarkovModel(BaseGenerativeModel):
    """Base class for HMM variants that share emission and transition structure.

    Literature: Rabiner (1989), Hamilton (1989). Concrete subclasses are
    SupervisedHmm, MarkovSwitchingHmm, SemiSupervisedHmm, and NeuralHmm.
    """

    name = "BaseHiddenMarkovModel"

    def Fit(self, train_data: pd.DataFrame, cfg: dict[str, Any] | None = None) -> Any:
        """
        Fit HMM parameters. Subclasses override with n_regimes.

        Parameters:
        train_data: pandas.DataFrame
            Frame with regime and emission columns.
        cfg: dict or None
            Pipeline config with an hmm block.

        Return:
           HMMFit.
        """
        raise NotImplementedError


class BaseDiffusionModel(BaseGenerativeModel):
    """Base class for DDPM return generators.

    Literature: Ho, Jain, and Abbeel (2020), Denoising Diffusion Probabilistic
    Models. UnconditionalDiffusion trains one generator on all windows.
    RegimeSpecialistDiffusion trains one generator per volatility regime.
    """

    name = "BaseDiffusionModel"

    def Fit(self, train_data: Any, cfg: dict[str, Any] | None = None) -> Any:
        """
        Train the diffusion generator.

        Parameters:
        train_data: any
            Window array or path to regime_*.npy.
        cfg: dict or None
            Pipeline config with a diffusion block.

        Return:
           Checkpoint path or trainer state.
        """
        raise NotImplementedError
