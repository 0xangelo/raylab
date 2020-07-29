# pylint:disable=missing-module-docstring
from raylab.agents.mapo import MAPOTorchPolicy
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import SPAML


class MlMAPOTorchPolicy(MAPOTorchPolicy):
    """PyTorch policy for MAPO-MLE.

    Effectively substitutes a Maximum Likelihood loss for the policy-aware model
    loss in MAPO.
    """

    # pylint:disable=too-many-ancestors

    @property
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.mapo.mle import MlMAPOTrainer

        return MlMAPOTrainer.options

    @property
    def model_training_loss(self):
        return self.loss_mle

    @property
    def model_warmup_loss(self):
        return self.loss_mle

    def _setup_model_loss(self):
        # Placeholder
        self.loss_paml = SPAML(
            self.module.models, self.module.actor, self.module.critics
        )

        self.loss_mle = ModelEnsembleMLE(self.module.models)

    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_actor.set_termination_fn(self.termination_fn)
