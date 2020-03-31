# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture(params=((), (1,), (2,)))
def sample_shape(request):
    return request.param
