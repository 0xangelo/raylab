"""Base implemenation for all NN policy modules."""
from typing import List

import numpy as np
from torch import nn


class Base(nn.Module):
    """Base class for all NN policy modules.

    Defines common boilerplate code and interfaces.
    """

    @staticmethod
    def get_initial_state() -> List[np.ndarray]:
        """Returns the initial state for recurrent modules.

        By default returns an empty list. Override this for actual recurrent modules.

        Warning:
            RLlib calls this method to initialize its networks. We don't need such
            initialization but implement this placeholder to avoid crashes.
        """
        return []
