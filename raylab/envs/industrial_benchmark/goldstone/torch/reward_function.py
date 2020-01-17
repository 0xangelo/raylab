"""PyTorch implementation of reward_function."""
import torch
import numpy as np

from .nlgp import TorchNLGP


class TorchRewardFunction:
    """
    generates the reward function for fixed phi
    works ONLY WITH SCALAR inputs
    Input:
     * phi: angle in Radians
     * max_required_step: the max. required step by the optimal policy; must be positive
    """

    def __init__(self, phi, max_required_step):
        self.phi = phi
        self.max_required_step = max_required_step
        if max_required_step <= 0:
            raise ValueError("Value for argument max_required_step must be positive")

        self._reward_function = self.reward_function_factory(phi, max_required_step)
        self.optimum_radius = self._compute_optimal_radius(phi, max_required_step)
        self.optimum_value = self._reward_function(self.optimum_radius)

    def reward(self, rad):
        """Apply the reward_function to the input."""
        return self._reward_function(rad)

    @staticmethod
    def _rad_transformation_factory(opt_rad, min_rad):
        def tsf(ten):
            exponent = (2 - opt_rad.abs()) / (2 - min_rad.abs())
            scaling = (2 - min_rad.abs()) / (2 - opt_rad.abs()) ** exponent
            return torch.where(
                ten.abs() <= opt_rad.abs(),
                ten * min_rad.abs() / opt_rad.abs(),
                ten.sign()
                * (min_rad.abs() + scaling * (ten.abs() - opt_rad.abs()) ** exponent),
            )

        return tsf

    @staticmethod
    def _compute_optimal_radius(phi, max_required_step):
        phi = phi % (2 * np.pi)

        opt = torch.max(phi.sign().abs(), torch.as_tensor(max_required_step))
        opt = torch.where(phi >= np.pi, opt.neg(), opt)
        return opt

    def reward_function_factory(self, phi, max_required_step):
        """
        Generate the reward function for fixed phi
        Input:
         * phi: angle in Radians
         * max_required_step: the max. required step by the optimal policy;
                              must be positive
        """
        nlgp = TorchNLGP()
        # use 2-pi-symmetry to move phi in domain [0,2pi]
        phi = phi % (2 * np.pi)
        # the desired radius at which we want the global optimim to be:
        opt_rad = self._compute_optimal_radius(phi, torch.as_tensor(max_required_step))
        #
        # the radius of minimum in NLGP:
        min_rad = nlgp.global_minimum_radius(phi)
        #
        # radius transformation such that minimum is moved to desired value
        trans = self._rad_transformation_factory(opt_rad, min_rad)
        return lambda r: nlgp.polar_nlgp(trans(r), phi)
