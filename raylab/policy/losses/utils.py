"""Common utilities for Loss functions."""
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import grad

from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict


def action_dpg(
    q_max: Tensor,
    a_max: Tensor,
    dqda_clipping: Optional[float] = None,
    clip_norm: bool = False,
) -> Tuple[Tensor, Tensor]:
    # pylint:disable=line-too-long
    """Deterministic policy gradient loss, similar to trfl.dpg.

    Inspired by `Acmes's DPG`_. Allows logging of Q-value action gradients.

    .. _`Acme's DPG`: https://github.com/deepmind/acme/blob/51c4db7c8ec27e040ac52d65347f6f4ecfe04f81/acme/tf/losses/dpg.py#L21

    Args:
        q_max: Q-value of the approximate greedy action of shape `(*,)`
        a_max: Action from the policy's output of shape `(*, A)`
        dqda_clipping: Optional value by which to clip the action gradients
        clip_norm: Whether to clip action grads by norm or value

    Returns:
        The DPG loss of shape `(*,)` and the norm of the action-value gradient
        of shape `(*,)`

    Note:
        This is a loss, so it's already supposed to be minimized, i.e., its
        gradients w.r.t. the policy parameters correspond to the negative DPG
    """
    # pylint:enable=line-too-long
    # Fake a Jacobian-vector product to calculate grads w.r.t. to batch of actions
    (dqda,) = grad(q_max, a_max, grad_outputs=torch.ones_like(q_max))
    # Keepdim to support broadcasting in clip_norm later
    dqda_norm = torch.norm(dqda, dim=-1, keepdim=True)

    if dqda_clipping:
        # pylint:disable=invalid-unary-operand-type
        if clip_norm:
            clip_coef = dqda_clipping / dqda_norm
            dqda = torch.where(clip_coef < 1, dqda * clip_coef, dqda)
        else:
            dqda = torch.clamp(dqda, min=-dqda_clipping, max=dqda_clipping)

    loss = -torch.sum(a_max * dqda.detach(), dim=-1)
    return loss, dqda_norm


def dist_params_stats(dist_params: TensorDict, name: str) -> StatDict:
    """Returns mean, max, and min for each distribution parameter.

    Args:
        dist_params: Dictionary mapping names to distribution parameters

    Returns:
        Dictionary with average, minimum, and maximum of each parameter as
        floats
    """
    items = tuple((k, v) for k, v in dist_params.items() if v.requires_grad)
    info = {}
    info.update({name + "/mean_" + k: v.mean().item() for k, v in items})
    info.update({name + "/max_" + k: v.max().item() for k, v in items})
    info.update({name + "/min_" + k: v.min().item() for k, v in items})
    return info
