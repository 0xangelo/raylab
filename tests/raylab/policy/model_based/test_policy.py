import itertools

import pytest
from ray.rllib.evaluation.metrics import get_learner_stats

from raylab.options import configure
from raylab.policy.model_based import MBPolicyMixin
from raylab.policy.off_policy import OffPolicyMixin


@pytest.fixture
def policy_cls(base_policy_cls):
    @configure
    @MBPolicyMixin.add_options
    @OffPolicyMixin.add_options
    class Policy(MBPolicyMixin, OffPolicyMixin, base_policy_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.build_replay_buffer()
            self.build_timers()
            self._epoch_seq = itertools.count()

        def train_dynamics_model(self, warmup: bool = False):
            return [], {"model_epochs": next(self._epoch_seq) + 1}

        def improve_policy(self, batch) -> dict:
            return {"improved": True}

    return Policy


@pytest.fixture(params=(1, 4), ids=lambda x: f"ModelInterval:{x}")
def model_update_interval(request):
    return request.param


@pytest.fixture
def config(model_update_interval):
    return {
        "policy": {
            "model_update_interval": model_update_interval,
            "module": {"type": "ModelBasedSAC"},
        }
    }


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(config=config)


def test_update_interval(policy, model_update_interval, samples):
    expected = 0

    for i in range(1, model_update_interval * 10 + 1):
        info = policy.learn_on_batch(samples)
        info = get_learner_stats(info)
        assert policy._learn_calls == i
        expected += 1 if (i % model_update_interval == 0 or i == 1) else 0
        assert info["model_epochs"] == expected
