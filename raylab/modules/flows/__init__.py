"""
Implements various flows.
Each flow is invertible so it can be forwarded (if training) and backwarded (if
evaluating).
Each flow also outputs its log det J "regularization"

General reference:
"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)

Mostly copied from
https://github.com/karpathy/pytorch-normalizing-flows
"""
from .abstract import NormalizingFlow, ConditionalNormalizingFlow
from .affine_constant import AffineConstantFlow, ActNorm
from .affine_half import Affine1DHalfFlow
from .cond_affine_half import CondAffine1DHalfFlow
from .maf import MAF, IAF


__all__ = [
    "ActNorm",
    "AffineConstantFlow",
    "Affine1DHalfFlow",
    "CondAffine1DHalfFlow",
    "ConditionalNormalizingFlow",
    "IAF",
    "MAF",
    "NormalizingFlow",
]
