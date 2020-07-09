import pytest
import torch


@pytest.fixture(scope="module", params=(True, False), ids=lambda x: f"Forked({x})")
def module_cls(request):
    from raylab.policy.modules.model.stochastic.ensemble import StochasticModelEnsemble
    from raylab.policy.modules.model.stochastic.ensemble import (
        ForkedStochasticModelEnsemble,
    )

    return ForkedStochasticModelEnsemble if request.param else StochasticModelEnsemble


@pytest.fixture(params=(1, 4), ids=lambda x: f"Ensemble({x})")
def ensemble_size(request):
    return request.param


@pytest.fixture
def build_single(obs_space, action_space):
    from raylab.policy.modules.model.stochastic.single import MLPModel

    spec = MLPModel.spec_cls()
    input_dependent_scale = True

    return lambda: MLPModel(obs_space, action_space, spec, input_dependent_scale)


@pytest.fixture
def module(module_cls, build_single, ensemble_size, torch_script):
    models = [build_single() for _ in range(ensemble_size)]

    module = module_cls(models)
    return torch.jit.script(module) if torch_script else module


def test_log_prob(module, log_prob_inputs, ensemble_size):
    obs = log_prob_inputs[0]
    log_prob = module.log_prob(*log_prob_inputs)

    assert torch.is_tensor(log_prob)
    assert log_prob.shape == (ensemble_size,) + obs.shape[:-1]
