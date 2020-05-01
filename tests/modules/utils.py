# pylint: disable=missing-docstring
from functools import partial

import torch
from ray.rllib.utils.tracking_dict import UsageTrackingDict

from raylab.utils.debug import fake_batch
from raylab.utils.pytorch import convert_to_tensor


def make_module(module_cls, obs_space, action_space, config, torch_script):
    mod = module_cls(obs_space, action_space, config)
    return torch.jit.script(mod) if torch_script else mod


def make_batch(obs_space, action_space, batch_size=4):
    batch = UsageTrackingDict(
        fake_batch(obs_space, action_space, batch_size=batch_size)
    )
    batch.set_get_interceptor(partial(convert_to_tensor, device="cpu"))
    return batch
