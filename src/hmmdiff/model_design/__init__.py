"""Model design and implementation.

BaseGenerativeModel / BaseHiddenMarkovModel / BaseDiffusionModel are the
abstract parents. SupervisedHmm and related classes inherit the HMM base.
UnconditionalDiffusion and RegimeSpecialistDiffusion inherit the diffusion
base. PathStitcher concatenates specialist draws along an HMM regime path.
"""

from hmmdiff.model_design.base_model import (
    BaseDiffusionModel,
    BaseGenerativeModel,
    BaseHiddenMarkovModel,
)
from hmmdiff.model_design.diffusion_trainer import (
    RegimeSpecialistDiffusion,
    UnconditionalDiffusion,
)
from hmmdiff.model_design.hmm_models import (
    HMMFit,
    MarkovSwitchingHmm,
    NeuralHmm,
    SemiSupervisedHmm,
    SupervisedHmm,
)
from hmmdiff.model_design.path_stitcher import PathStitcher

__all__ = [
    "BaseDiffusionModel",
    "BaseGenerativeModel",
    "BaseHiddenMarkovModel",
    "HMMFit",
    "MarkovSwitchingHmm",
    "NeuralHmm",
    "PathStitcher",
    "RegimeSpecialistDiffusion",
    "SemiSupervisedHmm",
    "SupervisedHmm",
    "UnconditionalDiffusion",
]
