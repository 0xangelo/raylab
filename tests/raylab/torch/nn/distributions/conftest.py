import pytest

SAMPLE_SHAPE = ((), (1,), (2,))


@pytest.fixture(params=SAMPLE_SHAPE, ids=(f"Sample{s}" for s in SAMPLE_SHAPE))
def sample_shape(request):
    return request.param
