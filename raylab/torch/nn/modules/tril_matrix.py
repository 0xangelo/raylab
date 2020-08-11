# pylint:disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils import override


class TrilMatrix(nn.Module):
    """Neural network module which outputs a lower-triangular matrix."""

    __constants__ = {"in_features", "matrix_dim", "row_sizes"}

    def __init__(self, in_features, matrix_dim):
        super().__init__()
        self.in_features = in_features
        self.matrix_dim = matrix_dim
        self.row_sizes = tuple(range(1, self.matrix_dim + 1))
        tril_dim = int(self.matrix_dim * (self.matrix_dim + 1) / 2)
        self.linear_module = nn.Linear(self.in_features, tril_dim)

    @override(nn.Module)
    def forward(self, logits):  # pylint:disable=arguments-differ
        # Batch of flattened lower triangular matrices: [..., N * (N + 1) / 2]
        flat_tril = self.linear_module(logits)
        # Split flat lower triangular into rows
        split_tril = torch.split(flat_tril, self.row_sizes, dim=-1)
        # Compute exponentiated diagonals, row by row
        tril_rows = []
        for row in split_tril:
            zeros = torch.zeros(row.shape[:-1] + (self.matrix_dim - row.shape[-1],))
            decomposed_row = [row[..., :-1], row[..., -1:] ** 2, zeros]
            tril_rows.append(torch.cat(decomposed_row, dim=-1))
        # Stack rows into a single (batched) matrix. dim=-2 ensures that we stack then
        # as rows, not columns (which would effectively transpose the matrix into an
        # upper triangular one)
        tril = torch.stack(tril_rows, dim=-2)
        return tril
