# pylint:disable=missing-docstring
# pylint: enable=missing-docstring
import torch.distributions as dists

from .transforms import TanhTransform


class SquashedDistribution(dists.TransformedDistribution):
    """Transformed distribution with samples constrained to the desired range."""

    # pylint:disable=abstract-method

    def __init__(self, base_distribution, low, high, **kwargs):
        squash = TanhTransform(cache_size=1)
        shift = dists.AffineTransform(
            loc=(high + low) / 2, scale=(high - low) / 2, cache_size=1, event_dim=1
        )
        super().__init__(base_distribution, [squash, shift], **kwargs)

    @property
    def mean(self):
        var = self.base_dist.mean
        for transform in self.transforms:
            var = transform(var)
        return var

    def reproduce(self, event):
        """Produce a reparametrized sample with the same value as `event`."""
        assert hasattr(
            self.base_dist, "reproduce"
        ), "Base distribution has no `reproduce` method"

        for transform in reversed(self.transforms):
            event = transform.inv(event)
        sample = self.base_dist.reproduce(event)
        for transform in self.transforms:
            sample = transform(sample)
        return sample
