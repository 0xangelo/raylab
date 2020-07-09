import pytest


@pytest.fixture(
    params=(pytest.param(True, marks=pytest.mark.slow), False),
    ids=("TorchScript", "Eager"),
    scope="module",
)
def torch_script(request):
    return request.param
