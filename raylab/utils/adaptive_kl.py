"""
From Proximal Policy Optimization Algorithms:
http://arxiv.org/abs/1707.06347
"""
from dataclasses import dataclass
from dataclasses import field


@dataclass
class AdaptiveKLCoeffSpec:
    """Adaptive schedule for KL Divergence regularization."""

    initial_coeff: float = 0.2
    desired_kl: float = 0.01
    adaptation_coeff: float = 2.0
    threshold: float = 1.5
    curr_coeff: float = field(init=False)

    def __post_init__(self):
        self.curr_coeff = self.initial_coeff

    def adapt(self, kl_div):
        """Apply PPO rule to update current KL coeff based on latest KL divergence."""
        if kl_div < self.desired_kl / self.threshold:
            self.curr_coeff /= self.adaptation_coeff
        elif kl_div > self.desired_kl * self.threshold:
            self.curr_coeff *= self.adaptation_coeff
