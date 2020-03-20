# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture(params=(True, False), ids=("TorchScript", "Eager"))
def torch_script(request):
    return request.param
