"""Policy for MAGE using PyTorch."""
from typing import List, Tuple

from nnrl.nn.critic import HardValue
from nnrl.optim import build_optimizer

from raylab.agents.sop import SOPTorchPolicy
from raylab.options import configure, option
from raylab.policy import EnvFnMixin
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import MAGE, MaximumLikelihood
from raylab.policy.model_based.lightning import LightningModelTrainer, TrainingSpec
from raylab.policy.model_based.policy import MBPolicyMixin
from raylab.utils.types import StatDict


def default_model_training() -> dict:
    """Model training routine used by MAGE paper."""
    spec = TrainingSpec()
    spec.datamodule.holdout_ratio = 0.0
    spec.datamodule.max_holdout = 0
    spec.datamodule.batch_size = 256
    spec.datamodule.shuffle = True
    spec.datamodule.num_workers = 0
    spec.training.max_epochs = None
    spec.training.max_steps = 120
    spec.training.patience = None
    spec.warmup = spec.training
    return spec.to_dict()


@configure
@option("model_training", default=default_model_training())
@option("model_update_interval", default=25)
@option("improvement_steps", default=10, override=True)
@option("policy_delay", 2, override=True)
@option("batch_size", default=1024, override=True)
@option("lambda", default=0.05, help="TD error regularization for MAGE loss")
@option("module/type", "MAGE", override=True)
@option("optimizer/models", default={"type": "Adam", "lr": 1e-4, "weight_decay": 1e-4})
@option("optimizer/actor", default={"type": "Adam", "lr": 1e-4}, override=True)
@option("optimizer/critics", default={"type": "Adam", "lr": 1e-4}, override=True)
@option("exploration_config/pure_exploration_steps", 1000, override=True)
class MAGETorchPolicy(MBPolicyMixin, EnvFnMixin, SOPTorchPolicy):
    """MAGE policy in PyTorch to use with RLlib.

    Attributes:
        loss_model: maximum likelihood loss for model ensemble
        loss_actor: deterministic policy gradient loss
        loss_critic: model-based action-value-gradient estimator loss
    """

    # pylint:disable=too-many-ancestors
    dist_class = WrapDeterministicPolicy
    model_trainer: LightningModelTrainer

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._set_model_loss()
        self._set_critic_loss()
        self.build_timers()
        self.model_trainer = LightningModelTrainer(
            models=self.module.models,
            loss_fn=self.loss_model,
            optimizer=self.optimizers["models"],
            replay=self.replay,
            config=self.config,
        )

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

    def train_dynamics_model(
        self, warmup: bool = False
    ) -> Tuple[List[float], StatDict]:
        return self.model_trainer.optimize(warmup=warmup)

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
        config = self.config["optimizer"]
        optimizers["models"] = build_optimizer(self.module.models, config["models"])
        return optimizers
