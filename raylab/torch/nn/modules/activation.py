"""Custom activation functions as neural network modules."""
import torch
import torch.nn as nn


class Swish(nn.Module):
    r"""Swish activation function.

    Notes:
        Applies the mapping :math:`x \mapsto x \cdot \sigma(x)`,
        where :math:`sigma` is the sigmoid function.

    Reference:
        Eger, Steffen, Paul Youssef, and Iryna Gurevych.
        "Is it time to swish? Comparing deep learning activation functions
        across NLP tasks."
        arXiv preprint arXiv:1901.02671 (2019).
    """
    # pylint:disable=arguments-differ,no-self-use

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * value.sigmoid()
