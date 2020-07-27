"""Common utilities for Loss functions."""
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import grad


def action_dpg(
    q_max: Tensor,
    a_max: Tensor,
    dqda_clipping: Optional[float] = None,
    clip_norm: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Deterministic policy gradient loss, similar to trfl.dpg.

    Args:
        q_max: Q-value of the approximate greedy action
        a_max: Action from the policy's output
        dqda_clipping: Optional value by which to clip the action gradients
        clip_norm: Whether to clip action grads by norm or value

    Returns:
        The DPG loss and the norm of the action-value gradient, both for
        each batch dimension

    Note:
        This is a loss, so it's already supposed to be minimized, i.e., its
        gradients w.r.t. the policy parameters correspond to the negative DPG
    """
    # Fake a Jacobian-vector product to calculate grads w.r.t. to batch of actions
    dqda = grad(q_max, [a_max], grad_outputs=torch.ones_like(q_max))[0]
    dqda_norm = torch.norm(dqda, dim=-1, keepdim=True)

    if dqda_clipping:
        # pylint:disable=invalid-unary-operand-type
        if clip_norm:
            clip_coef = dqda_clipping / dqda_norm
            dqda = torch.where(clip_coef < 1, dqda * clip_coef, dqda)
        else:
            dqda = torch.clamp(dqda, min=-dqda_clipping, max=dqda_clipping)

    # Target_a ensures correct gradient calculated during backprop.
    target_a = dqda + a_max
    # Stop the gradient going through Q network when backprop.
    target_a = target_a.detach()
    # Gradient only go through actor network.
    loss = 0.5 * torch.sum(torch.square(target_a - a_max), dim=-1)
    # This recovers the DPG because (letting w be the actor network weights):
    # d(loss)/dw = 0.5 * (2 * (target_a - a_max) * d(target_a - a_max)/dw)
    #            = (target_a - a_max) * [d(target_a)/dw  - d(a_max)/dw]
    #            = dq/da * [d(target_a)/dw  - d(a_max)/dw]  # by defn of target_a
    #            = dq/da * [0 - d(a_max)/dw]                # by stop_gradient
    #            = - dq/da * da/dw
    return loss, dqda_norm
