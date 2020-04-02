"""Base classes for Normalizing Flows."""
from ..distributions import Transform, ConditionalTransform


class NormalizingFlow(Transform):
    """A diffeomorphism.

    Flows are specialized `Transform`s with tractable Jacobians. They can be used
    in most situations where a `Transform` would be (e.g., with `ComposeTransform`).
    All flows map samples from a latent space to another (f(z) -> x)
    Use the `reverse` flag to invert the flow (f^{-1}(x) -> z).
    """


class ConditionalNormalizingFlow(ConditionalTransform):
    """A Normalizing Flow conditioned on some external variable."""
