# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.registry import AGENTS

TRAINER_NAMES, TRAINER_IMPORTS = zip(*AGENTS.items())


@pytest.fixture(scope="module", params=TRAINER_IMPORTS, ids=TRAINER_NAMES)
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy
