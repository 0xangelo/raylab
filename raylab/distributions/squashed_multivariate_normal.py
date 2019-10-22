# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch.distributions as dists

from .transforms import TanhTransform
from .diag_multivariate_normal import DiagMultivariateNormal


class SquashedMultivariateNormal(dists.TransformedDistribution):
    """
    Multivariate Normal distribution with diagonal covariance matrix and constrained
    to the desired range.
    """

    # pylint: disable=abstract-method

    def __init__(self, loc, scale_diag, low, high, **kwargs):
        base_distribution = DiagMultivariateNormal(loc, scale_diag, **kwargs)
        squash = TanhTransform(cache_size=1)
        shift = dists.AffineTransform(
            loc=(high + low) / 2, scale=(high - low) / 2, cache_size=1, event_dim=1
        )
        super().__init__(base_distribution, [squash, shift])

    @property
    def mean(self):
        var = self.base_dist.mean
        for transform in self.transforms:
            var = transform(var)
        return var
