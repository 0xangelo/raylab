"""
Hessian-free optimization utilities
"""
import torch
from torch.autograd import grad


def hessian_vector_product(output, params, vector):
    """Computes the Hessian vector product w.r.t to a scalar loss.

    Args:
        output (Tensor): loss tensor w.r.t. which the Hessian will be computed.
        params (list): the parameters of the module, usually from a call to
            `module.parameters()`.
        vector (Tensor): The flattened vector to compute the Hessian product with.
            This must have the same total number of elements in `params`.
    """
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
