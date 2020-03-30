"""
Hessian-free optimization utilities
"""
import torch

from raylab.utils.pytorch import flat_grad


def fisher_vec_prod(vec, obs, action, policy, damping=1e-3):
    """Approximately compute the fisher-vector-product using samples."""
    cur_logp = policy.log_prob(obs, action)
    avg_kl = torch.mean(cur_logp - cur_logp.detach())
    grad = flat_grad(avg_kl, policy.parameters(), create_graph=True)
    fvp = flat_grad(grad.dot(vec), policy.parameters()).detach()
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
