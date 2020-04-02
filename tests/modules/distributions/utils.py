# pylint:disable=missing-docstring
import torch


def _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape):
    # pylint:disable=too-many-arguments
    sample, _ = dist.sample(params, sample_shape)
    assert sample.shape == sample_shape + batch_shape + event_shape

    rsample, _ = dist.rsample(params, sample_shape)
    if not torch.isnan(rsample).any():
        assert rsample.shape == sample_shape + batch_shape + event_shape
        rsample_ = dist.reproduce(params, rsample.requires_grad_())
        if not torch.isnan(rsample_).any():
            assert torch.allclose(rsample_, rsample)

    log_prob = dist.log_prob(params, sample)
    assert log_prob.shape == sample_shape + batch_shape

    entropy = dist.entropy(params)
    if not torch.isnan(entropy).any():
        assert entropy.shape == batch_shape
        perplexity = dist.perplexity(params)
        assert perplexity.shape == entropy.shape

    cdf = dist.cdf(params, sample)
    if not torch.isnan(cdf).any():
        assert cdf.shape == sample_shape + batch_shape + event_shape
        icdf = dist.icdf(params, cdf)
        if not torch.isnan(icdf).any():
            assert icdf.shape == cdf.shape
