# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def policy_and_batch_(obs_space, action_space):
    def make_policy_and_batch(policy_cls, config):
        policy = policy_cls(obs_space, action_space, config)
        batch = policy._lazy_tensor_dict(
            fake_batch(obs_space, action_space, batch_size=10)
        )
        return policy, batch

    return make_policy_and_batch
