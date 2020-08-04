import pytest
import torch


@pytest.fixture(scope="module", params=(True, False), ids=lambda x: f"Forked({x})")
def module_cls(request):
    from raylab.policy.modules.model.stochastic.ensemble import SME
    from raylab.policy.modules.model.stochastic.ensemble import ForkedSME

    return ForkedSME if request.param else SME


@pytest.fixture(params=(1, 4), ids=lambda x: f"Ensemble({x})")
def ensemble_size(request):
    return request.param


@pytest.fixture
def expand_foreach_model(ensemble_size):
    def expand(tensor):
        return [tensor.clone() for _ in range(ensemble_size)]

    return expand


@pytest.fixture
def build_single(obs_space, action_space):
    from raylab.policy.modules.model.stochastic.single import MLPModel

    spec = MLPModel.spec_cls(standard_scaler=True, input_dependent_scale=True)

    return lambda: MLPModel(obs_space, action_space, spec)


@pytest.fixture
def module(module_cls, build_single, ensemble_size, torch_script):
    models = [build_single() for _ in range(ensemble_size)]

    module = module_cls(models)
    return torch.jit.script(module) if torch_script else module


def test_forward(module, obs, act, expand_foreach_model):
    obs, act = map(expand_foreach_model, (obs, act))

    params = module(obs, act)
    assert all(["loc" in p for p in params])
    assert all(["scale" in p for p in params])
    assert all(["min_logvar" in p for p in params])
    assert all(["max_logvar" in p for p in params])

    assert all([(p["scale"].log() > p["min_logvar"]).all() for p in params])
    assert all([(p["scale"].log() < p["max_logvar"]).all() for p in params])


def test_iterate(module, ensemble_size):
    assert len([1 for m in module]) == ensemble_size


def test_log_prob(module, obs, act, next_obs, rew, expand_foreach_model):
    # pylint:disable=too-many-arguments
    obs, act, next_obs = map(expand_foreach_model, (obs, act, next_obs))
    log_prob = module.log_prob(obs, act, next_obs)

    assert isinstance(log_prob, list)
    assert all([torch.is_tensor(logp) for logp in log_prob])
    assert all([logp.shape == rew.shape for logp in log_prob])

    def check_grad(grad):
        return grad is None or torch.allclose(grad, torch.zeros_like(grad))

    all_params = set(module.parameters())
    for idx, logp in enumerate(log_prob):
        for par in all_params:
            par.grad = None

        logp.mean().backward(retain_graph=True)
        single_params = set(module[idx].parameters())
        assert all(not check_grad(p.grad) for p in single_params)
        assert all(check_grad(p.grad) for p in all_params - single_params)


def test_sample(module, obs, act, rew, expand_foreach_model):
    obs, act = map(expand_foreach_model, (obs, act))

    outputs = module.sample(obs, act)
    assert isinstance(outputs, list)
    assert all([isinstance(o, tuple) for o in outputs])
    assert all([torch.is_tensor(s) and torch.is_tensor(p) for s, p in outputs])

    samples, logp = zip(*outputs)
    samples_, _ = zip(*module.sample(obs, act))

    assert all([s.shape == o.shape for s, o in zip(samples, obs)])
    assert all([s.dtype == o.dtype for s, o in zip(samples, obs)])
    assert all([p.shape == rew.shape for p in logp])
    assert all([p.dtype == rew.dtype for p in logp])
    assert all([not torch.allclose(s, s_) for s, s_ in zip(samples, samples_)])


def test_rsample_from_params(module, obs, act, rew, expand_foreach_model):
    obs, act = map(expand_foreach_model, (obs, act))

    params = module(obs, act)
    outputs = module.rsample_from_params(params)
    assert isinstance(outputs, list)
    assert all([isinstance(o, tuple) for o in outputs])
    assert all([torch.is_tensor(s) and torch.is_tensor(p) for s, p in outputs])
    assert all([s.shape == obs[0].shape for s, _ in outputs])
    assert all([p.shape == rew.shape for _, p in outputs])
