"""Mathematical functions."""
import torch


def cbrt(value):
    """Cube root. Equivalent to torch.pow(value, 1/3), but numerically stable.

    Source: https://github.com/bayesiains/nsf/blob/master/utils/torchutils.py
    """
    return torch.sign(value) * torch.exp(torch.log(torch.abs(value)) / 3.0)
