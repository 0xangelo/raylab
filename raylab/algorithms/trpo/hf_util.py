"""
Hessian-free optimization utilities
"""
import torch
from torch.autograd import grad


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
    fvp = -hessian_vector_product(cur_logp.mean(), policy.parameters(), vec)
    return fvp + vec * damping


def hessian_vector_product(output, params, vector):
    # pylint:disable=missing-docstring
    params = list(params)
    vecs, idx = [], 0
    for par in params:
        vecs += [vector[idx : idx + par.numel()].reshape_as(par)]
        idx += par.numel()
    grads = grad(output, params, allow_unused=True, create_graph=True)
    hvp = grad([g for g in grads if g is not None], params, vecs, allow_unused=True)
    zeros = torch.zeros
    return torch.cat(
        [(zeros(p.numel()) if v is None else v.flatten()) for v, p in zip(hvp, params)]
    )


def conjugate_gradient(f_mat_vec_prod, b, cg_iters=10, residual_tol=1e-6):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b,
    where we only have access to f: x -> Ax
    """
    # pylint:disable=invalid-name
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
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

    return x, i + 1, rdotr


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
