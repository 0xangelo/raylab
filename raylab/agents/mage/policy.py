"""Policy for MAGE using PyTorch."""
from raylab.agents.sop import SOPTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import MAGE
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses.mage import MAGEModules
from raylab.pytorch.optim import build_optimizer


class MAGETorchPolicy(ModelTrainingMixin, EnvFnMixin, SOPTorchPolicy):
    """MAGE policy in PyTorch to use with RLlib.

    Attributes:
        loss_model: maximum likelihood loss for model ensemble
        loss_actor: deterministic policy gradient loss
        loss_critic: model-based action-value-gradient estimator loss
    """

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        module = self.module
        self.loss_model = ModelEnsembleMLE(module.models)
        mage_modules = MAGEModules(
            critics=module.critics,
            target_critics=module.target_critics,
            policy=module.actor,
            target_policy=module.target_actor,
            models=module.models,
        )
        self.loss_critic = MAGE(mage_modules)
        self.loss_critic.gamma = self.config["gamma"]
        self.loss_critic.lambda_ = self.config["lambda"]

    @staticmethod
    def get_default_config():
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.mage import MAGETrainer

        return MAGETrainer._default_config

    def _set_reward_hook(self):
        self.loss_critic.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_critic.set_termination_fn(self.termination_fn)

    def make_optimizers(self):
        optimizers = super().make_optimizers()
        optimizers["models"] = build_optimizer(
            self.module.models, self.config["torch_optimizer"]["models"]
        )
        return optimizers

    def compile(self):
        super().compile()
        for loss in (self.loss_model, self.loss_actor, self.loss_critic):
            loss.compile()
