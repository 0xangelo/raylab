# pylint: disable=missing-docstring,redefined-outer-name,protected-access,invalid-name
import pytest
import numpy as np
import torch


BATCH_SIZE = 32


@pytest.fixture
def env(hvac_env):
    return hvac_env({})


def test_observation_space(env):
    obs_space = env.observation_space
    assert obs_space is not None
    assert obs_space.low.shape == (env._num_rooms + 1,)
    assert obs_space.high.shape == (env._num_rooms + 1,)
    assert np.allclose(obs_space.low[:-1], [-np.inf] * env._num_rooms)
    assert np.allclose(obs_space.low[-1], 0.0)
    assert np.allclose(obs_space.high[:-1], [np.inf] * env._num_rooms)
    assert np.allclose(obs_space.high[-1], 1.0)

    obs = obs_space.sample()
    assert obs is not None
    assert obs in obs_space


def test_action_space(env):
    action_space = env.action_space
    assert action_space is not None
    assert action_space.low.shape == (env._num_rooms,)
    assert action_space.high.shape == (env._num_rooms,)
    assert np.allclose(action_space.low, 0.0)
    assert np.allclose(action_space.high, 1.0)

    action = action_space.sample()
    assert action is not None
    assert action in action_space


def test_reset(env):
    state = env.reset()
    assert state in env.observation_space


def test_temp_outside(env):
    sample, logp = env._temp_outside()
    assert sample.shape == (env._num_rooms,)
    assert logp.shape == (env._num_rooms,)

    sample, logp = env._temp_outside(sample_shape=(BATCH_SIZE,))
    assert sample.shape == (BATCH_SIZE, env._num_rooms)
    assert logp.shape == (BATCH_SIZE, env._num_rooms)


def test_temp_hall(env):
    sample, logp = env._temp_hall()
    assert sample.shape == (env._num_rooms,)
    assert logp.shape == (env._num_rooms,)

    sample, logp = env._temp_hall(sample_shape=(BATCH_SIZE,))
    assert sample.shape == (BATCH_SIZE, env._num_rooms)
    assert logp.shape == (BATCH_SIZE, env._num_rooms)


def test_temp(env):
    action = env.action_space.sample()
    AIR_MAX = torch.as_tensor(env._config["AIR_MAX"])
    action = torch.as_tensor(action) * AIR_MAX

    SAMPLE_SHAPE = ()
    temp_hall, _ = env._temp_hall(SAMPLE_SHAPE)
    temp_outside, _ = env._temp_outside(SAMPLE_SHAPE)

    temp = env._temp(action, temp_outside, temp_hall)
    assert temp.shape == (env._num_rooms,)


def test_transition_fn(env):
    state = env.observation_space.sample()
    action = env.action_space.sample()
    next_state, logp = env.transition_fn(state, action)

    assert next_state.numpy() in env.observation_space
    assert next_state.shape == (env._num_rooms + 1,)
    assert logp.shape == (env._num_rooms,)

    next_state, logp = env.transition_fn(state, action, sample_shape=(BATCH_SIZE,))
    assert np.all(s.numpy() in env.observation_space for s in next_state)
    assert next_state.shape == (BATCH_SIZE, env._num_rooms + 1)
    assert logp.shape == (BATCH_SIZE, env._num_rooms)


def test_reward_fn(env):
    state = env.observation_space.sample()
    action = env.action_space.sample()
    next_state, _ = env.transition_fn(state, action)

    reward = env.reward_fn(state, action, next_state)
    assert reward.shape == ()
    assert (reward <= 0.0).all()

    next_state, _ = env.transition_fn(state, action, sample_shape=(BATCH_SIZE,))
    reward = env.reward_fn(state, action, next_state)
    assert reward.shape == (BATCH_SIZE,)
    assert (reward <= 0.0).all()


def test_step(env):
    state = env.reset()
    assert state in env.observation_space

    for _ in range(env._horizon - 1):
        action = env.action_space.sample()
        assert action in env.action_space

        state, reward, done, info = env.step(action)
        assert state in env.observation_space
        assert reward <= 0.0
        assert not done
        assert info == {}

    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    assert done


def test_unpack_state(env):
    obs = env.observation_space.sample()
    state, time = env._unpack_state(obs)
    assert state.shape == (env._num_rooms,)
    assert time.shape == ()
