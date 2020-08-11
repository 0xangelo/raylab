# pylint:disable=missing-docstring
# pylint: enable=missing-docstring
import torch
import torch.distributions as ptd


class Logistic(ptd.TransformedDistribution):
    """Creates a Logistic distribution parameterized by loc and scale."""

    # pylint:disable=abstract-method

    def __init__(self, loc, scale, **kwargs):
        loc, scale = map(torch.as_tensor, (loc, scale))
        base_distribution = ptd.Uniform(
            torch.zeros_like(loc), torch.ones_like(loc), **kwargs
        )
        transforms = [
            ptd.SigmoidTransform().inv,
            ptd.AffineTransform(loc=loc, scale=scale),
        ]
        super().__init__(base_distribution, transforms)
