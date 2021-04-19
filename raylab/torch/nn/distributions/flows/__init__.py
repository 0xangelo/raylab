"""Flow implementations.

Each flow is invertible so it can be computed (if training) and inverted
(if evaluating). Each flow also outputs its log of the Jacobian determinant.

References:
    Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling
    and Inference." arXiv preprint arXiv:1912.02762 (2019).
"""
from . import masks, utils
from .abstract import (
    CompositeTransform,
    ConditionalTransform,
    InverseTransform,
    Transform,
)
from .affine_constant import ActNorm, AffineConstantFlow
from .coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    CouplingTransform,
    PiecewiseRQSCouplingTransform,
)
from .maf import IAF, MAF
from .nonlinearities import (
    AffineTransform,
    SigmoidSquashTransform,
    SigmoidTransform,
    TanhSquashTransform,
    TanhTransform,
)

__all__ = [
    "ActNorm",
    "AdditiveCouplingTransform",
    "AffineConstantFlow",
    "AffineCouplingTransform",
    "AffineTransform",
    "ConditionalTransform",
    "CouplingTransform",
    "IAF",
    "InverseTransform",
    "MAF",
    "masks",
    "PiecewiseRQSCouplingTransform",
    "SigmoidTransform",
    "SigmoidSquashTransform",
    "TanhTransform",
    "TanhSquashTransform",
    "Transform",
]
