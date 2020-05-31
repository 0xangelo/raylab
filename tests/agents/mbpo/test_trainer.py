# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.utils.debug import fake_batch


REAL_RATIO = (0.0, 0.5, 1.0)


@pytest.fixture(
    scope="module", params=REAL_RATIO, ids=(f"ReadData%({r})" for r in REAL_RATIO)
)
def real_data_ratio(request):
    return request.param


@pytest.fixture
def config(real_data_ratio):
    return {"real_data_ratio": real_data_ratio}


def test_improve_policy(trainer_cls, envs, config):
    # pylint:disable=unused-argument
    trainer = trainer_cls(env="MockEnv", config=config)
    env = trainer.workers.local_worker().env

    real_samples = fake_batch(env.observation_space, env.action_space, batch_size=80)
    for row in real_samples.rows():
        trainer.replay.add(row)
    virtual_samples = fake_batch(
        env.observation_space, env.action_space, batch_size=800
    )
    for row in virtual_samples.rows():
        trainer.virtual_replay.add(row)

    info = trainer.improve_policy(1)
    assert "learner" not in info
    assert "learner_stats" not in info
