"""Continuous Q-Learning with Normalized Advantage Functions."""
import time

from ray import tune
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.agents.trainer import Trainer, with_common_config

from raylab.utils.replay_buffer import ReplayBuffer
from raylab.algorithms.naf.naf_policy import NAFTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {"layers": [400, 300], "activation": "ELU", "ortho_init_gain": 0.2},
        # Which type of TorchScript to use with the module. Possible types include
        # None: module is not converted to TorchScript
        # trace: module is converted using torch.jit.trace
        # script: module is converted using torch.jit.script
        "torch_script": None,
        # === Optimization ===
        # Name of Pytorch optimizer class
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the PyTorch optimizer
        "torch_optimizer_options": {"lr": 1e-3},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Rollout Worker ===
        "num_workers": 0,
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # diag_gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        # full_gaussian: use gaussian action space noise where the precision matrix is
        #     given by the advantage function P matrix
        "exploration": None,
        # Whether to act greedly or exploratory, mostly for evaluation purposes
        "greedy": False,
        # Scaling term of the lower triangular matrix for the multivariate gaussian
        # action distribution
        "scale_tril_coeff": 1.0,
        # Gaussian stddev for diagonal gaussian action space noise
        "diag_gaussian_stddev": 0.1,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 1000,
        # Options for parameter noise exploration
        "param_noise_spec": {
            "initial_stddev": 0.1,
            "desired_action_stddev": 0.2,
            "adaptation_coefficient": 1.01,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"greedy": True, "pure_exploration_steps": 0},
    }
)


class NAFTrainer(Trainer):
    """Single agent trainer for NAF."""

    # pylint: disable=attribute-defined-outside-init

    _name = "NAF"
    _default_config = DEFAULT_CONFIG
    _policy = NAFTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        # Dummy optimizer to log stats
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(config["buffer_size"])

    @override(Trainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        start = time.time()
        steps_sampled = 0
        while True:
            self.update_exploration_phase()

            samples = worker.sample()
            steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

            if (
                time.time() - start >= self.config["min_iter_time_s"]
                and steps_sampled >= self.config["timesteps_per_iteration"]
            ):
                break

        self.optimizer.num_steps_sampled += steps_sampled

        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=steps_sampled,
            info=dict(learner=stats, **res.get("info", {})),
        )
        return res

    # === New Methods ===

    def update_exploration_phase(self):
        """Signal to policies if training is still in the pure exploration phase."""
        global_timestep = self.optimizer.num_steps_sampled
        pure_expl_steps = self.config["pure_exploration_steps"]
        if pure_expl_steps:
            only_explore = global_timestep < pure_expl_steps
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, _: p.set_pure_exploration_phase(only_explore)
            )

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."

        # Taken from ray.rllib.agents.dqn.dqn
        if config["exploration"] == "parameter_noise":
            if config["batch_mode"] != "complete_episodes":
                raise ValueError(
                    "Exploration with parameter space noise requires "
                    "batch_mode to be complete_episodes."
                )

            if config["callbacks"]["on_episode_start"]:
                start_callback = config["callbacks"]["on_episode_start"]
            else:
                start_callback = None

            def on_episode_start(info):
                # as a callback function to sample and pose parameter space
                # noise on the parameters of network
                policies = info["policy"]
                for pol in policies.values():
                    pol.perturb_policy_parameters()
                if start_callback:
                    start_callback(info)

            config["callbacks"]["on_episode_start"] = tune.function(on_episode_start)
