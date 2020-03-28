# pylint:disable=missing-docstring


def _test_dist_ops(
    dist, params, batch_shape, event_shape, sample_shape, test_icdf=True
):
    # pylint:disable=too-many-arguments
    sample = dist.sample(params, sample_shape)
    rsample = dist.rsample(params, sample_shape)
    assert sample.shape == sample_shape + batch_shape + event_shape
    assert rsample.shape == sample_shape + batch_shape + event_shape

    log_prob = dist.log_prob(params, sample)
    assert log_prob.shape == sample_shape + batch_shape
    cdf = dist.cdf(params, sample)
    assert cdf.shape == sample_shape + batch_shape
    entropy = dist.entropy(params)
    assert entropy.shape == batch_shape
    perplexity = dist.perplexity(params)
    assert perplexity.shape == entropy.shape

    if test_icdf:
        icdf = dist.icdf(params, cdf)
        assert icdf.shape == sample.shape
