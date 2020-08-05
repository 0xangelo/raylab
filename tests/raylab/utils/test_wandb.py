import pytest

CONFIGS = ({"wandb": {}}, {}, {"wandb": {"entity": "test", "project": "dummy"}})


pytest.skip(allow_module_level=True)


@pytest.fixture(params=CONFIGS, ids=lambda x: f"Create:{bool(x.get('wandb'))}")
def config(request):
    return request.param


@pytest.fixture
def logger(config):
    from raylab.utils.wandb import WandBLogger

    log = WandBLogger(config, name="DUMMY")
    yield log
    log.stop()


def test_init(logger, config):
    if config.get("wandb"):
        assert logger.enabled
    else:
        assert not logger.enabled
