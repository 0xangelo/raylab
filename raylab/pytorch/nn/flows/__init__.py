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

from . import masks
from .abstract import ConditionalTransform
from .abstract import Transform
from .affine_constant import ActNorm
from .affine_constant import AffineConstantFlow
from .coupling import AdditiveCouplingTransform
from .coupling import AffineCouplingTransform
from .coupling import CouplingTransform
from .coupling import PiecewiseRQSCouplingTransform
from .maf import IAF
from .maf import MAF


__all__ = [
    "ActNorm",
    "AdditiveCouplingTransform",
    "AffineConstantFlow",
    "AffineCouplingTransform",
    "ConditionalTransform",
    "CouplingTransform",
    "IAF",
    "MAF",
    "masks",
    "PiecewiseRQSCouplingTransform",
    "Transform",
]
