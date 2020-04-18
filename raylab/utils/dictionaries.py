"""Utilities for manipulating python dictionaries and general mappings."""
import copy

from ray.rllib.utils import deep_update


def deep_merge(dict1, dict2, *args, **kwargs):
    """Deep copy original dict and pass it to RLlib's deep_update."""
    clone = copy.deepcopy(dict1)
    return deep_update(clone, dict2, *args, **kwargs)
