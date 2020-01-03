# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import numpy as np
import torch


BATCH_SIZE = 32


@pytest.fixture
def env(reservoir_env):
    return reservoir_env({})


@pytest.fixture(params=((), (1,), (2,)))
def sample_shape(request):
    return request.param


def get_batch_states(env):
    return torch.as_tensor(
        np.stack([env.observation_space.sample() for _ in range(BATCH_SIZE)])
    )


def get_batch_actions(env):
    return torch.as_tensor(
        np.stack([env.action_space.sample() for _ in range(BATCH_SIZE)])
    )


def test_observation_space(env):
    obs_space = env.observation_space
    assert obs_space is not None
    assert obs_space.low.shape == (env._num_reservoirs + 1,)
    assert obs_space.high.shape == (env._num_reservoirs + 1,)
    assert np.allclose(obs_space.low, 0.0)
    assert np.allclose(obs_space.high[:-1], np.inf)
    assert np.allclose(obs_space.high[-1], 1.0)

    obs = obs_space.sample()
    assert obs is not None
    assert obs in obs_space


def test_action_space(env):
    action_space = env.action_space
    assert action_space is not None
    assert action_space.low.shape == (env._num_reservoirs,)
    assert action_space.high.shape == (env._num_reservoirs,)
    assert np.allclose(action_space.low, 0.0)
    assert np.allclose(action_space.high, 1.0)

    action = action_space.sample()
    assert action is not None
    assert action in action_space


def test_reset(env):
    state = env.reset()
    assert state in env.observation_space


def test_rainfall(env, sample_shape):
    rlevel, _ = env._unpack_state(env.observation_space.sample())
    rain1, logp1 = env._rainfall(rlevel, sample_shape)
    assert torch.all(rain1 >= 0.0)
    assert rain1.shape == sample_shape + (env._num_reservoirs,)
    assert logp1.shape == sample_shape

    sample_shape = (128,)
    rain2, logp2 = env._rainfall(rlevel, sample_shape=sample_shape)
    assert torch.all(rain2 >= 0.0)
    assert rain2.shape == sample_shape + (env._num_reservoirs,)
    assert logp2.shape == sample_shape


def test_evaporated(env):
    obs = env.observation_space.sample()
    state, _ = env._unpack_state(obs)
    evaporated = env._evaporated(state)
    assert torch.all(evaporated >= 0.0)
    assert evaporated.shape == state.shape


def test_overflow(env):
    state, _ = env._unpack_state(env.observation_space.sample())
    action = env.action_space.sample()
    overflow = env._overflow(state, action)
    assert overflow.shape == state.shape
    assert torch.all(overflow >= 0.0)


def test_inflow(env):
    state, _ = env._unpack_state(env.observation_space.sample())
    action = torch.as_tensor(env.action_space.sample())
    inflow = env._inflow(state, action)
    assert inflow.shape == state.shape


def test_rlevel(env):
    max_res_cap = torch.as_tensor(env._config["MAX_RES_CAP"])
    state, _ = env._unpack_state(env.observation_space.sample())
    action = torch.as_tensor(env.action_space.sample())
    rain, _ = env._rainfall(state)
    rlevel = env._rlevel(state, action, rain)
    assert rlevel.shape == state.shape
    assert torch.all(rlevel >= 0.0)
    assert torch.all(rlevel <= max_res_cap)


@pytest.fixture(params=(True, False))
def tensorfy(request):
    return request.param


def test_transition_fn(env, tensorfy, sample_shape):
    state = env.observation_space.sample()
    action = env.action_space.sample()
    if tensorfy:
        state, action = map(torch.as_tensor, (state, action))

    next_state, logp = env.transition_fn(state, action, sample_shape=sample_shape)
    assert (
        next_state.numpy() if sample_shape == () else next_state[0].numpy()
    ) in env.observation_space
    assert next_state.shape == sample_shape + (env._num_reservoirs + 1,)
    assert logp.shape == sample_shape


def test_reward_fn(env):
    next_state = get_batch_states(env)
    reward = env.reward_fn(None, None, next_state)
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
    assert state.shape == (env._num_reservoirs,)
    assert time.shape == (1,)
