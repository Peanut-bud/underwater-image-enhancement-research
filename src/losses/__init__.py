"""Loss stacks for supervised and adaptation stages."""

from .perceptual import VGGPerceptualLoss
from .supervised_losses import SupervisedLossStack, SupervisedLossWeights
from .unsupervised_losses import AdaptationLossStack, AdaptationLossWeights

__all__ = [
    "AdaptationLossStack",
    "AdaptationLossWeights",
    "SupervisedLossStack",
    "SupervisedLossWeights",
    "VGGPerceptualLoss",
]
