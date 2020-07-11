import pytest
import torch


@pytest.fixture(
    params=(pytest.param(True, marks=pytest.mark.slow), False),
    ids=("TorchScript", "Eager"),
    scope="module",
)
def torch_script(request):
    return request.param


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    from raylab.utils.debug import fake_batch

    samples = fake_batch(obs_space, action_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}
