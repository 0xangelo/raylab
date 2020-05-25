"""Utilities for computing losses."""
import torch


def clipped_action_value(obs, actions, critics):
    """Compute the minimum of two action-value functions on state-action pairs."""
    value, _ = torch.cat([m(obs, actions) for m in critics], dim=-1).min(dim=-1)
    return value
