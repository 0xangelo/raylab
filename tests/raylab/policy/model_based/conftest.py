# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import BaseActionDist


@pytest.fixture(scope="module")
def action_dist():
    class ActionDist(BaseActionDist):
        # pylint:disable=abstract-method
        @classmethod
        def _check_model_compat(cls, *args, **kwargs):
            pass

    return ActionDist


@pytest.fixture(scope="module")
def base_policy_cls(action_dist, obs_space, action_space):
    class Policy(TorchPolicy):
        # pylint:disable=abstract-method
        dist_class = action_dist

        def __init__(self, config):
            super().__init__(obs_space, action_space, config)

    return Policy
