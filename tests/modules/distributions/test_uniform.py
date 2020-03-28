# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch

from raylab.modules.distributions import Uniform


def test_uniform(torch_script):
    # pylint:disable=unused-variable
    dist = torch.jit.script(Uniform()) if torch_script else Uniform()
    flat = torch.stack([torch.zeros([]), torch.ones([])], dim=0)

    params = dist(flat)
    assert "low" in params
    assert "high" in params

    sample = dist.sample(params)
    rsample = dist.rsample(params)
    log_prob = dist.log_prob(params, sample)
    cdf = dist.cdf(params, sample)
    icdf = dist.icdf(params, cdf)
    entropy = dist.entropy(params)
    perplexity = dist.perplexity(params)
