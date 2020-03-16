# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import TrilMatrix


@pytest.fixture(params=(1, 4, 10))
def in_features(request):
    return request.param


@pytest.fixture(params=(1, 4, 10))
def matrix_dim(request):
    return request.param


def test_tril_matrix(in_features, matrix_dim, torch_script):
    maker = TrilMatrix.as_script_module if torch_script else TrilMatrix
    module = maker(in_features, matrix_dim)

    inputs = torch.randn(1, in_features)
    module(inputs)
