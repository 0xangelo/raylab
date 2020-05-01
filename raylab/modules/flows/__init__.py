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
from .abstract import Transform, ConditionalTransform
from .affine_constant import AffineConstantFlow, ActNorm
from .coupling import (
    AffineCouplingTransform,
    AdditiveCouplingTransform,
    CouplingTransform,
    PiecewiseRQSCouplingTransform,
)
from .maf import MAF, IAF
from . import masks


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
