# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
from typing import Dict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils import override

from .abstract import ConditionalDistribution


class Categorical(ConditionalDistribution):
    r"""
    Creates a categorical distribution parameterized by `logits`.

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from
        :math:`\{0, \ldots, K-1\}` where `K` is ``logits.size(-1)``.

    Example::

        >>> params = {"logits": torch.tensor([0.25, 0.25, 0.25, 0.25])}
        >>> m = Categorical()
        >>> m.sample(params)  # equal probability of 0, 1, 2, 3
        tensor(3)
    """

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return {"logits": inputs - inputs.logsumexp(dim=-1, keepdim=True)}

    @override(ConditionalDistribution)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        logits = self._unpack_params(params)
        params_shape = sample_shape + logits.shape
        sample_shape = sample_shape + logits.shape[:-1]
        probs = F.softmax(logits, dim=-1).expand(params_shape)
        probs_2d = probs.reshape(-1, logits.shape[-1])
        sample_2d = torch.multinomial(probs_2d, 1, True)
        out = sample_2d.reshape(sample_shape)
        return out, self.log_prob(out, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def log_prob(self, value, params: Dict[str, torch.Tensor]):
        logits = self._unpack_params(params)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    @override(ConditionalDistribution)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        logits = self._unpack_params(params)
        probs = F.softmax(logits, dim=-1)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    @override(ConditionalDistribution)
    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        logits = self._unpack_params(params)
        sample = torch.argmax(logits, dim=-1)
        return sample, self.log_prob(sample, params)

    def _unpack_params(self, params: Dict[str, torch.Tensor]):
        # pylint:disable=no-self-use
        return params["logits"]
