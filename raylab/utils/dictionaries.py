"""Utilities for manipulating python dictionaries and general mappings."""
import copy
from typing import Optional

from ray.rllib.utils import deep_update


def deep_merge(
    dict1,
    dict2,
    new_keys_allowed: bool = False,
    allow_new_subkey_list: Optional[list] = None,
    override_all_if_type_changes: Optional[list] = None,
):
    """Deep copy original dict and pass it to RLlib's deep_update."""
    clone = copy.deepcopy(dict1)
    return deep_update(
        clone,
        dict2,
        new_keys_allowed=new_keys_allowed,
        allow_new_subkey_list=allow_new_subkey_list,
        override_all_if_type_changes=override_all_if_type_changes,
    )


def get_keys(mapping, *keys):
    """Return the values corresponding to the given keys, in order."""
    return (mapping[k] for k in keys)


def all_except(mapping, *exclude):
    """Return a new mapping with all keys except `exclude`.

    Keys must be hashable to be used with `set`.
    """
    exclude = set(exclude)
    return {k: v for k, v in mapping.items() if k not in exclude}
