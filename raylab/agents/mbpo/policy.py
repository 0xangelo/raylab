"""Policy for MBPO using PyTorch."""
from typing import List
from typing import Tuple

from ray.rllib import SampleBatch

from raylab.agents.sac import SACTorchPolicy
from raylab.options import configure
from raylab.options import option
from raylab.policy import learner_stats
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import MaximumLikelihood
from raylab.policy.model_based import EnvFnMixin
from raylab.policy.model_based import LightningModelTrainer
from raylab.policy.model_based import ModelSamplingMixin
from raylab.policy.model_based.policy import MBPolicyMixin
from raylab.policy.model_based.policy import model_based_options
from raylab.policy.model_based.sampling import SamplingSpec
from raylab.torch.optim import build_optimizer
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.timer import TimerStat
from raylab.utils.types import StatDict


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
@model_based_options
@LightningModelTrainer.add_options
@option(
    "virtual_buffer_size",
    default=int(1e6),
    help="Size of the buffer for virtual samples",
)
@option(
    "model_rollouts",
    default=40,
    help="""Number of model rollouts to add to virtual buffer each policy interval.

    Populates virtual replay with this many model rollouts before each policy
    improvement.
    """,
)
@option(
    "real_data_ratio",
    default=0.1,
    help="Fraction of each policy minibatch to sample from environment replay pool",
)
@option("module", default=DEFAULT_MODULE, override=True)
@option(
    "optimizer/models",
    default={"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
)
@option("model_sampling", default=SamplingSpec().to_dict(), help=SamplingSpec.__doc__)
class MBPOTorchPolicy(MBPolicyMixin, EnvFnMixin, ModelSamplingMixin, SACTorchPolicy):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=too-many-ancestors
    virtual_replay: NumpyReplayBuffer
    model_trainer: LightningModelTrainer
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        models = self.module.models
        self.loss_model = MaximumLikelihood(models)

        self.build_timers()
        self.model_trainer = LightningModelTrainer(
            models=self.module.models,
            loss_fn=self.loss_model,
            optimizer=self.optimizers["models"],
            replay=self.replay,
            config=self.config,
        )

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["optimizer"]
        optimizers["models"] = build_optimizer(self.module.models, config["models"])
        return optimizers

    def build_replay_buffer(self):
        super().build_replay_buffer()
        self.virtual_replay = NumpyReplayBuffer(
            self.observation_space,
            self.action_space,
            self.config["virtual_buffer_size"],
        )
        self.virtual_replay.seed(self.config["seed"])

    def build_timers(self):
        super().build_timers()
        self.timers["augmentation"] = TimerStat()

    @learner_stats
    def learn_on_batch(self, samples: SampleBatch) -> dict:
        self.add_to_buffer(samples)
        self._learn_calls += 1

        info = {}
        warmup = self._learn_calls == 1
        if self._learn_calls % self.config["model_update_interval"] == 0 or warmup:
            with self.timers["model"] as timer:
                losses, model_info = self.train_dynamics_model(warmup=warmup)
                timer.push_units_processed(model_info["model_epochs"])
                info.update(model_info)
            self.set_new_elite(losses)

        with self.timers["augmentation"] as timer:
            count_before = len(self.virtual_replay)
            self.populate_virtual_buffer()
            timer.push_units_processed(len(self.virtual_replay) - count_before)

        with self.timers["policy"] as timer:
            times = self.config["improvement_steps"]
            policy_info = self.update_policy(times=times)
            timer.push_units_processed(times)
            info.update(policy_info)

        info.update(self.timer_stats())
        return info

    def train_dynamics_model(
        self, warmup: bool = False
    ) -> Tuple[List[float], StatDict]:
        return self.model_trainer.optimize(warmup=warmup)

    def populate_virtual_buffer(self):
        # pylint:disable=missing-function-docstring
        num_rollouts = self.config["model_rollouts"]
        real_data_ratio = self.config["real_data_ratio"]
        if not (num_rollouts and real_data_ratio < 1.0):
            return

        real_samples = self.replay.sample(num_rollouts)
        virtual_samples = self.generate_virtual_sample_batch(real_samples)
        self.virtual_replay.add(virtual_samples)

    def update_policy(self, times: int) -> StatDict:
        batch_size = self.config["batch_size"]
        env_batch_size = int(batch_size * self.config["real_data_ratio"])
        model_batch_size = batch_size - env_batch_size

        for _ in range(times):
            samples = []
            if env_batch_size:
                samples += [self.replay.sample(env_batch_size)]
            if model_batch_size:
                samples += [self.virtual_replay.sample(model_batch_size)]
            batch = SampleBatch.concat_samples(samples)
            batch = self.lazy_tensor_dict(batch)
            info = self.improve_policy(batch)

        return info

    def timer_stats(self) -> dict:
        stats = super().timer_stats()
        augmentation_timer = self.timers["augmentation"]
        stats.update(
            augmentation_time_s=round(augmentation_timer.mean, 3),
            augmentation_throughput=round(augmentation_timer.mean_throughput, 3),
        )
        return stats
