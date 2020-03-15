"""Registry of modules for PyTorch policies."""
from .naf_module import NAFModule

MODULES = {"NAFModule": NAFModule}


def get_module(name, obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    return MODULES[name](obs_space, action_space, config)
