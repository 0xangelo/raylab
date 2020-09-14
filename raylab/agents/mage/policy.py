"""Policy for MAGE using PyTorch."""
from raylab.agents.sop import SOPTorchPolicy
from raylab.options import configure
from raylab.options import option
from raylab.policy import EnvFnMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import MAGE
from raylab.policy.losses import MaximumLikelihood
from raylab.policy.model_based.training import TrainingSpec
from raylab.policy.modules.critic import HardValue
from raylab.torch.optim import build_optimizer


@configure
@option("lambda", default=0.05, help="TD error regularization for MAGE loss")
@option("model_training", default=TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@option(
    "model_warmup",
    default=TrainingSpec().to_dict(),
    help="""Specifications for model warm-up.

    Same configurations as 'model_training'.
    """,
)
@option("module/type", "ModelBasedDDPG", override=True)
@option("torch_optimizer/models", {"type": "Adam"})
class MAGETorchPolicy(ModelTrainingMixin, EnvFnMixin, SOPTorchPolicy):
    """MAGE policy in PyTorch to use with RLlib.

    Attributes:
        loss_model: maximum likelihood loss for model ensemble
        loss_actor: deterministic policy gradient loss
        loss_critic: model-based action-value-gradient estimator loss
    """

    # pylint:disable=too-many-ancestors
    dist_class = WrapDeterministicPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._set_model_loss()
        self._set_critic_loss()

    def build_replay_buffer(self):
        pass

    def _set_model_loss(self):
        self.loss_model = MaximumLikelihood(self.module.models)

    def _set_critic_loss(self):
        module = self.module
        target_critic = HardValue(
            policy=module.target_actor, q_value=module.target_critics
        )
        self.loss_critic = MAGE(
            critics=module.critics,
            policy=module.actor,
            target_critic=target_critic,
            models=module.models,
        )
        self.loss_critic.gamma = self.config["gamma"]
        self.loss_critic.lambd = self.config["lambda"]

    def learn_on_batch(self, samples):
        batch = self.lazy_tensor_dict(samples)
        info = self.improve_policy(batch)
        info.update(self.get_exploration_info())
        return info

    @property
    def model_training_loss(self):
        return self.loss_model

    def compile(self):
        super().compile()
        for loss in (self.loss_model, self.loss_actor, self.loss_critic):
            loss.compile()

    def _set_reward_hook(self):
        self.loss_critic.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_critic.set_termination_fn(self.termination_fn)

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers["models"] = build_optimizer(
            self.module.models, self.config["torch_optimizer"]["models"]
        )
        return optimizers
