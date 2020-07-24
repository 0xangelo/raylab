# pylint:disable=missing-module-docstring
from raylab.agents.mapo import MAPOTorchPolicy
from raylab.policy.losses import ModelEnsembleMLE


class MlMAPOTorchPolicy(MAPOTorchPolicy):
    """PyTorch policy for MAPO-MLE.

    Effectively substitutes a Maximum Likelihood loss for the policy-aware model
    loss in MAPO.
    """

    # pylint:disable=abstract-method

    @property
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.mapo.mle import MlMAPOTrainer

        return MlMAPOTrainer.options

    def _setup_model_loss(self):
        self.loss_model = ModelEnsembleMLE(self.module.models)

    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_actor.set_termination_fn(self.termination_fn)
