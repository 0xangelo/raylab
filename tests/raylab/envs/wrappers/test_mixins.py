import pytest

import raylab.envs.wrappers as wrappers

RNG_SUBCLASSES = {
    wrappers.CorrelatedIrrelevant,
    wrappers.LinearRedundant,
    wrappers.RandomIrrelevant,
}


@pytest.fixture(params=RNG_SUBCLASSES, ids=lambda x: type(x).__name__)
def wrapper_cls(request):
    return request.param
