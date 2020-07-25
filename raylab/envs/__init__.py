"""OpenAI Gym environments and utilities."""
from ray.tune import register_env

from .rewards import get_reward_fn
from .rewards import has_reward_fn
from .rewards import register as register_reward_fn
from .termination import get_termination_fn
from .termination import has_termination_fn
from .termination import register as register_termination_fn
from .utils import get_env_creator
from .utils import has_env_creator

__all__ = [
    "get_reward_fn",
    "has_reward_fn",
    "register_reward_fn",
    "get_termination_fn",
    "has_termination_fn",
    "register_termination_fn",
    "register_env",
    "get_env_creator",
    "has_env_creator",
]
