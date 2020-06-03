"""Flow implementations.

Each flow is invertible so it can be computed (if training) and inverted
(if evaluating). Each flow also outputs its log of the Jacobian determinant.

References:
    Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling
    and Inference." arXiv preprint arXiv:1912.02762 (2019).
"""

from . import masks
from . import utils
from .abstract import CompositeTransform
from .abstract import ConditionalTransform
from .abstract import InverseTransform
from .abstract import Transform
from .affine_constant import ActNorm
from .affine_constant import AffineConstantFlow
from .coupling import AdditiveCouplingTransform
from .coupling import AffineCouplingTransform
from .coupling import CouplingTransform
from .coupling import PiecewiseRQSCouplingTransform
from .maf import IAF
from .maf import MAF
from .nonlinearities import AffineTransform
from .nonlinearities import SigmoidSquashTransform
from .nonlinearities import SigmoidTransform
from .nonlinearities import TanhSquashTransform
from .nonlinearities import TanhTransform


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
