"""Collection of custom RLlib Policy classes."""
from raylab.policy.torch_policy import TorchPolicy
from raylab.policy.kl_coeff_mixin import AdaptiveKLCoeffMixin
from raylab.policy.target_networks_mixin import TargetNetworksMixin
