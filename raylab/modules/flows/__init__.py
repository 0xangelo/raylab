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
from .abstract import ComposeNormalizingFlow, NormalizingFlowModel
from .affine_constant import AffineConstantFlow, ActNorm
from .affine_half import AffineHalfFlow
from .cond_affine_half import CondAffineHalfFlow
