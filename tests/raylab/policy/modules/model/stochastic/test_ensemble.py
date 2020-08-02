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
        return tensor.expand((ensemble_size,) + tensor.shape)

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
    assert "loc" in params
    assert "scale" in params
    assert "min_logvar" in params
    assert "max_logvar" in params

    assert (params["scale"].log() > params["min_logvar"]).all()
    assert (params["scale"].log() < params["max_logvar"]).all()


def test_iterate(module, ensemble_size):
    assert len([1 for m in module]) == ensemble_size


def test_log_prob(module, obs, act, next_obs, rew, expand_foreach_model):
    # pylint:disable=too-many-arguments
    obs, act, next_obs, rew = map(expand_foreach_model, (obs, act, next_obs, rew))
    log_prob = module.log_prob(obs, act, next_obs)

    assert torch.is_tensor(log_prob)
    assert log_prob.shape == rew.shape

    def check_grad(grad):
        return grad is None or torch.allclose(grad, torch.zeros_like(grad))

    all_params = set(module.parameters())
    for idx, logp in enumerate(log_prob.split(1, dim=0)):
        for par in all_params:
            par.grad = None

        logp.mean().backward(retain_graph=True)
        single_params = set(module[idx].parameters())
        assert all(not check_grad(p.grad) for p in single_params)
        assert all(check_grad(p.grad) for p in all_params - single_params)


def test_sample(module, obs, act, rew, expand_foreach_model):
    obs, act, rew = map(expand_foreach_model, (obs, act, rew))

    samples, logp = module.sample(obs, act)
    samples_, _ = module.sample(obs, act)
    assert samples.shape == obs.shape
    assert samples.dtype == obs.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)
