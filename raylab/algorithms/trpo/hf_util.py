"""
Hessian-free optimization utilities
"""
import torch

from raylab.utils.pytorch import flat_grad


def fisher_vec_prod(vec, obs, policy, n_samples=1, damping=1e-3):
    """Approximately compute the fisher-vector-product using samples.

    This is based on the Fisher Matrix formulation as the expected hessian
    of the negative log likelihood. For more information, see:
    https://en.wikipedia.org/wiki/Fisher_information#Matrix_form

    Args:
        vec (Tensor): The vector to compute the Fisher vector product with.
        obs (Tensor): The observations to evaluate the policy in.
        policy (nn.Module): The policy
        n_samples (int): The number of actions to sample for each state.
        damping (float): Regularization to prevent the Fisher from becoming singular.
    """
    _, cur_logp = policy.sample(obs, sample_shape=(n_samples,))
    grad = flat_grad(
        cur_logp.mean(), policy.parameters(), create_graph=True, allow_unused=True
    )
    fvp = -flat_grad(grad.dot(vec), policy.parameters(), allow_unused=True)
    return fvp + vec * damping


def conjugate_gradient(f_mat_vec_prod, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b,
    where we only have access to f: x -> Ax
    """
    # pylint:disable=invalid-name
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_mat_vec_prod(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


@torch.no_grad()
def line_search(
    func,
    x_0,
    d_x,
    expected_improvement,
    y_0=None,
    accept_ratio=0.1,
    backtrack_ratio=0.8,
    max_backtracks=15,
    atol=1e-7,
):
    """Perform a linesearch on func with start x_0 and direction d_x."""
    # pylint:disable=too-many-arguments
    if y_0 is None:
        y_0 = func(x_0)

    if expected_improvement >= atol:
        for exp in range(max_backtracks):
            ratio = backtrack_ratio ** exp
            x_new = x_0 - ratio * d_x
            y_new = func(x_new)
            improvement = y_0 - y_new
            # Armijo condition
            if improvement / (expected_improvement * ratio) >= accept_ratio:
                return x_new, expected_improvement * ratio, improvement

    return x_0, expected_improvement, 0
