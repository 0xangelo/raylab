"""NAF policy class using PyTorch."""
import inspect
import itertools

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as torch_util
import raylab.utils.param_noise as param_noise
import raylab.algorithms.naf.naf_module as modules
from raylab.policy import TorchPolicy


class NAFTorchPolicy(TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        self.optimizer = self._make_optimizer(self.module, self.config)

        # Flag for uniform random actions
        self._pure_exploration = False
        if self.config["exploration"] == "parameter_noise":
            self._param_noise_spec = param_noise.AdaptiveParamNoiseSpec(
                **config["param_noise_spec"]
            )

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @torch.no_grad()
    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

        if self.config["greedy"]:
            actions = self._greedy_actions(obs_batch)
        elif self._pure_exploration:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["exploration"] == "full_gaussian":
            actions = self._multivariate_gaussian_actions(obs_batch)
        elif self.config["exploration"] == "diag_gaussian":
            actions = self._diagonal_gaussian_actions(obs_batch)
        else:
            actions = self.module["policy"](obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        if self.config["exploration"] == "parameter_noise":
            self.update_parameter_noise(sample_batch)
        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        loss, info = self.compute_loss(batch_tensors, self.module, self.config)
        self.optimizer.zero_grad()
        loss.backward()

        total_norm = 0
        for param in self.module["naf"].parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        info["grad_norm"] = total_norm

        self.optimizer.step()

        torch_util.update_polyak(
            self.module["value"], self.module["target_value"], self.config["polyak"]
        )
        return {LEARNER_STATS_KEY: info}

    # === NEW METHODS ===

    # === Exploration ===
    def set_pure_exploration_phase(self, phase):
        """Set a boolean flag that tells the policy to act randomly."""
        self._pure_exploration = phase

    def perturb_policy_parameters(self):
        """Update the perturbed policy's parameters for exploration."""
        torch_util.perturb_module_params(
            self.module["policy"],
            self.module["target_policy"],
            self._param_noise_spec.stddev,
        )

    def update_parameter_noise(self, sample_batch):
        """Update parameter noise stddev given a batch from the perturbed policy."""
        noisy_actions = sample_batch[SampleBatch.ACTIONS]
        target_actions = self.module["target_policy"](
            self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        ).numpy()
        distance = param_noise.ddpg_distance_metric(noisy_actions, target_actions)
        self._param_noise_spec.adapt(distance)

    # === Action Sampling ===
    def _greedy_actions(self, obs_batch):
        policy = self.module["policy"]
        if "target_policy" in self.module:
            policy = self.module["target_policy"]

        out = policy(obs_batch)
        if isinstance(out, tuple):
            out, _ = out
        return out

    def _uniform_random_actions(self, obs_batch):
        dist = torch.distributions.Uniform(
            self.convert_to_tensor(self.action_space.low),
            self.convert_to_tensor(self.action_space.high),
        )
        actions = dist.sample(sample_shape=obs_batch.shape[:-1])
        return actions

    def _multivariate_gaussian_actions(self, obs_batch):
        loc, scale_tril = self.module["policy"](obs_batch)
        scale_coeff = self.config["scale_tril_coeff"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=scale_tril * scale_coeff
        )
        actions = dist.sample()
        return actions

    def _diagonal_gaussian_actions(self, obs_batch):
        loc = self.module["policy"](obs_batch)
        stddev = self.config["diag_gaussian_stddev"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=torch.diag(torch.ones(self.action_space.shape)) * stddev
        )
        actions = dist.sample()
        return actions

    # === Static Methods ===

    @staticmethod
    def compute_loss(batch_tensors, module, config):
        """Compute the forward pass of NAF's loss function.

        Arguments:
            batch_tensors (UsageTrackingDict): Dictionary of experience batches that are
                lazily converted to tensors.
            module (nn.Module): The module of the policy
            config (dict): The policy's configuration

        Returns:
            A scalar tensor sumarizing the losses for this experience batch.
        """
        with torch.no_grad():
            next_value = module["target_value"](batch_tensors[SampleBatch.NEXT_OBS])
            gamma = config["gamma"]
            target_value = torch.where(
                batch_tensors[SampleBatch.DONES],
                batch_tensors[SampleBatch.REWARDS],
                batch_tensors[SampleBatch.REWARDS] + gamma * next_value.squeeze(-1),
            )
            if config["clipped_double_q"]:
                twin_next_value = module["twin_target_value"](
                    batch_tensors[SampleBatch.NEXT_OBS]
                )
                twin_target_value = torch.where(
                    batch_tensors[SampleBatch.DONES],
                    batch_tensors[SampleBatch.REWARDS],
                    batch_tensors[SampleBatch.REWARDS]
                    + gamma * twin_next_value.squeeze(-1),
                )
                target_value = torch.min(target_value, twin_target_value)

        action_value = module["naf"](
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        td_error = torch.nn.MSELoss()(action_value.squeeze(-1), target_value)
        if config["clipped_double_q"]:
            twin_action_value = module["twin_naf"](
                batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
            )
            twin_td_error = torch.nn.MSELoss()(
                twin_action_value.squeeze(-1), target_value
            )
            td_error = (td_error + twin_td_error) / 2

        stats = {
            "q_mean": action_value.mean().item(),
            "q_max": action_value.max().item(),
            "q_min": action_value.min().item(),
            "td_error": td_error.item(),
        }
        return td_error, stats

    @staticmethod
    def _make_module(obs_space, action_space, config):
        # Create base modules
        logits_module_kwargs = dict(
            in_features=obs_space.shape[0],
            units=config["module"]["layers"],
            activation=config["module"]["activation"],
            layer_norm=(config["exploration"] == "parameter_noise"),
        )

        def action_module_kwargs(in_features):
            return dict(
                in_features=in_features,
                action_low=torch.from_numpy(action_space.low).float(),
                action_high=torch.from_numpy(action_space.high).float(),
            )

        logits_module = modules.FullyConnectedModule(**logits_module_kwargs)
        value_module = modules.ValueModule(logits_module.out_features)
        action_module = modules.ActionModule(
            **action_module_kwargs(logits_module.out_features)
        )
        tril_module = modules.TrilMatrixModule(
            logits_module.out_features, action_space.shape[0]
        )
        advantage_module = modules.AdvantageModule(tril_module, action_module)

        # Build main components
        module = nn.ModuleDict()
        module["naf"] = modules.NAF(logits_module, value_module, advantage_module)
        module["value"] = nn.Sequential(logits_module, value_module)
        module["target_value"] = nn.Sequential(
            modules.FullyConnectedModule(**logits_module_kwargs),
            modules.ValueModule(logits_module.out_features),
        )
        module["target_value"].load_state_dict(module["value"].state_dict())

        if config["clipped_double_q"]:
            twin_logits_module = modules.FullyConnectedModule(**logits_module_kwargs)
            twin_value_module = modules.ValueModule(twin_logits_module.out_features)
            module["twin_naf"] = modules.NAF(
                twin_logits_module,
                twin_value_module,
                modules.AdvantageModule(
                    modules.TrilMatrixModule(
                        twin_logits_module.out_features, action_space.shape[0]
                    ),
                    modules.ActionModule(
                        **action_module_kwargs(twin_logits_module.out_features)
                    ),
                ),
            )
            module["twin_value"] = nn.Sequential(twin_logits_module, twin_value_module)
            module["twin_target_value"] = nn.Sequential(
                modules.FullyConnectedModule(**logits_module_kwargs),
                modules.ValueModule(twin_logits_module.out_features),
            )
            module["twin_target_value"].load_state_dict(
                module["twin_value"].state_dict()
            )

        # Configure policy module based on exploration strategy
        if config["exploration"] == "full_gaussian":
            module["policy"] = modules.MultivariateGaussianPolicy(
                logits_module, action_module, tril_module
            )
        elif config["exploration"] == "parameter_noise":
            module["policy"] = modules.DeterministicPolicy(
                logits_module=modules.FullyConnectedModule(**logits_module_kwargs),
                action_module=modules.ActionModule(
                    **action_module_kwargs(logits_module.out_features)
                ),
            )
            module["target_policy"] = modules.DeterministicPolicy(
                logits_module, action_module
            )
        else:
            module["policy"] = modules.DeterministicPolicy(logits_module, action_module)

        # Initialize modules
        module.apply(
            torch_util.initialize_orthogonal(config["module"]["ortho_init_gain"])
        )

        if config["torch_script"] == "trace":
            trace_components(module, obs_space, action_space)
        elif config["torch_script"] == "script":
            script_components(module)
        return module

    @staticmethod
    def _make_optimizer(module, config):
        optimizer = config["torch_optimizer"]
        if isinstance(optimizer, str):
            optimizer_cls = torch_util.get_optimizer_class(optimizer)
        elif inspect.isclass(optimizer):
            optimizer_cls = optimizer
        else:
            raise ValueError(
                "'torch_optimizer' must be a string or class, got '{}'".format(
                    type(optimizer)
                )
            )

        optimizer_options = config["torch_optimizer_options"]
        parameters = module["naf"].parameters()
        if config["clipped_double_q"]:
            parameters = itertools.chain(parameters, module["twin_naf"].parameters())
        optimizer = optimizer_cls(parameters, **optimizer_options)
        return optimizer


def trace_components(module, obs_space, action_space):
    """Use tracing to produce TorchScript modules."""
    obs = torch.randn(1, *obs_space.shape)
    actions = torch.randn(1, *action_space.shape)
    module["naf"] = torch.jit.trace(module["naf"], (obs, actions))
    module["value"] = torch.jit.trace(module["value"], obs)
    module["target_value"] = torch.jit.trace(module["target_value"], obs)
    module["policy"] = torch.jit.trace(module["policy"], obs)
    if "twin_naf" in module:
        module["twin_naf"] = torch.jit.trace(module["twin_naf"], (obs, actions))
        module["twin_value"] = torch.jit.trace(module["twin_value"], obs)
        module["twin_target_value"] = torch.jit.trace(module["twin_target_value"], obs)


def script_components(module):
    """Use scripting to produce TorchScript modules."""
    module["naf"] = torch.jit.script(module["naf"])
    module["value"] = torch.jit.script(module["value"])
    module["target_value"] = torch.jit.script(module["target_value"])
    module["policy"] = torch.jit.script(module["policy"])
    if "twin_naf" in module:
        module["twin_naf"] = torch.jit.script(module["twin_naf"])
        module["twin_value"] = torch.jit.script(module["twin_value"])
        module["twin_target_value"] = torch.jit.script(module["twin_target_value"])
