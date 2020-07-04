# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from raylab.policy import OptimizerCollection


@pytest.fixture
def optimizer():
    params = [nn.Parameter(torch.empty([]))]
    return Adam(params)


@pytest.fixture
def collection():
    return OptimizerCollection()


def test_init(collection):
    assert not collection
    assert not list(collection.keys())
    assert not list(collection.values())
    assert not list(collection.items())


def test_set_invalid_obj(collection):
    with pytest.raises(ValueError):
        collection["invalid"] = []


def test_setitem(collection, optimizer):
    collection["param"] = optimizer

    assert "param" in collection
    assert len(collection) == 1
    assert list(collection) == ["param"]

    del collection["param"]
    assert "param" not in collection
    assert not collection
    assert not list(collection)
