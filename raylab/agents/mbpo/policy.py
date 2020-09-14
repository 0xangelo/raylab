"""Policy for MBPO using PyTorch."""
from ray.rllib import SampleBatch

from raylab.agents.sac import SACTorchPolicy
from raylab.options import configure
from raylab.options import option
from raylab.policy import EnvFnMixin
from raylab.policy import ModelSamplingMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import MaximumLikelihood
from raylab.policy.model_based.sampling import SamplingSpec
from raylab.policy.model_based.training import TrainingSpec
from raylab.torch.optim import build_optimizer


DEFAULT_MODULE = {
    "type": "ModelBasedSAC",
    "model": {
        "network": {"units": (128, 128), "activation": "Swish"},
        "ensemble_size": 7,
        "input_dependent_scale": True,
        "parallelize": True,
        "residual": True,
    },
    "actor": {
        "encoder": {"units": (128, 128), "activation": "Swish"},
        "input_dependent_scale": True,
    },
    "critic": {
        "double_q": True,
        "encoder": {"units": (128, 128), "activation": "Swish"},
    },
    "entropy": {"initial_alpha": 0.05},
}


@configure
@option("module", default=DEFAULT_MODULE, override=True)
@option(
    "torch_optimizer/models",
    default={"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
)
@option("model_training", default=TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@option(
    "model_warmup",
    default=TrainingSpec().to_dict(),
    help="""Specifications for model warm-up.

    Same configurations as 'model_training'.
    """,
)
@option("model_sampling", default=SamplingSpec().to_dict(), help=SamplingSpec.__doc__)
class MBPOTorchPolicy(
    EnvFnMixin, ModelTrainingMixin, ModelSamplingMixin, SACTorchPolicy
):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=too-many-ancestors
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        models = self.module.models
        self.loss_model = MaximumLikelihood(models)

    def build_replay_buffer(self):
        pass

    def learn_on_batch(self, samples: SampleBatch) -> dict:
        batch = self.lazy_tensor_dict(samples)
        info = self.improve_policy(batch)
        info.update(self.get_exploration_info())
        return info

    @property
    def model_training_loss(self):
        return self.loss_model

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]
        optimizers["models"] = build_optimizer(self.module.models, config["models"])
        return optimizers
