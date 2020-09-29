import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def is_batch(batch):
    def builder(is_key: str) -> dict:
        bat = batch.copy()
        bat[is_key] = torch.randn_like(batch[SampleBatch.REWARDS])
        return bat

    return builder
