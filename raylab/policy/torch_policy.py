"""Base for all PyTorch policies."""
from abc import abstractmethod
import contextlib
import io

import torch
from ray.tune.logger import pretty_print
from ray.rllib.models.model import restore_original_dimensions, flatten
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_ops import convert_to_non_torch_type
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib import SampleBatch

from raylab.agents import Trainer
from raylab.modules.catalog import get_module
from raylab.utils.dictionaries import deep_merge
from raylab.utils.pytorch import convert_to_tensor
from .action_dist import WrapModuleDist


class TorchPolicy(Policy):
    """Custom TorchPolicy that aims to be more general than RLlib's one."""

    def __init__(self, observation_space, action_space, config):
        self.framework = "torch"
        config = deep_merge(
            {**self.get_default_config(), "worker_index": None},
            config,
            new_keys_allowed=False,
            whitelist=Trainer._allow_unknown_subkeys,
            override_all_if_type_changes=Trainer._override_all_subkeys_if_type_changes,
        )
        super().__init__(observation_space, action_space, config)

        self.exploration = self._create_exploration()
        # Already set in Policy, might as well use it
        self.dist_class = WrapModuleDist

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.module = self.make_module(observation_space, action_space, self.config)
        self.module.to(self.device)
        self.optimizer = self.make_optimizer()

    @staticmethod
    @abstractmethod
    def get_default_config():
        """Return the default config for this policy class."""

    @abstractmethod
    def make_optimizer(self):
        """PyTorch optimizer to use."""

    @staticmethod
    def make_module(obs_space, action_space, config):
        """Build the PyTorch nn.Module to be used by this policy.

        Arguments:
            obs_space (gym.spaces.Space): the observation space for this policy
            action_space (gym.spaces.Space): the action_space for this policy
            config (dict): the user config containing the 'module' key

        Returns:
            A neural network module.
        """
        return get_module(obs_space, action_space, config["module"])

    @torch.no_grad()
    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs
    ):
        # pylint:disable=too-many-arguments,too-many-locals
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        input_dict = self._lazy_tensor_dict(
            {SampleBatch.CUR_OBS: obs_batch, "is_training": False}
        )
        if prev_action_batch:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        state_batches = [self.convert_to_tensor(s) for s in (state_batches or [])]

        # Call the exploration before_compute_actions hook.
        self.exploration.before_compute_actions(timestep=timestep)

        dist_inputs, state_out = self.compute_module_output(
            self._unpack_observations(input_dict),
            state_batches,
            self.convert_to_tensor([1]),
        )

        action_dist = self.dist_class(dist_inputs, self.module)
        actions, logp = self.exploration.get_exploration_action(
            action_distribution=action_dist, timestep=timestep, explore=explore
        )
        input_dict[SampleBatch.ACTIONS] = actions

        # Add default and custom fetches.
        extra_fetches = self.extra_action_out(
            input_dict, state_batches, self.module, action_dist
        )

        if logp is not None:
            prob, logp = map(convert_to_non_torch_type, (logp.exp(), logp))
            extra_fetches[SampleBatch.ACTION_PROB] = prob
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        return convert_to_non_torch_type((actions, state_out, extra_fetches))

    def _unpack_observations(self, input_dict):
        restored = input_dict.copy()
        restored["obs"] = restore_original_dimensions(
            input_dict["obs"], self.observation_space, self.framework
        )
        if len(input_dict["obs"].shape) > 2:
            restored["obs_flat"] = flatten(input_dict["obs"], self.framework)
        else:
            restored["obs_flat"] = input_dict["obs"]
        return restored

    def compute_module_output(self, input_dict, state, seq_lens):
        """Call the module with the given input tensors and state.

        This mirrors the method used by RLlib to execute the forward pass. Nested
        observation tensors are unpacked before this function is called.

        Subclasses should override this for custom forward passes (e.g., for recurrent
        networks).

        Arguments:
            input_dict (dict): dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training"
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths
        """
        # pylint:disable=unused-argument,no-self-use
        return {"obs": input_dict["obs"]}, state

    @torch.no_grad()
    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        # pylint:disable=too-many-arguments
        input_dict = self._lazy_tensor_dict(
            {SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions}
        )
        if prev_action_batch:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch

        dist_inputs, _ = self.module(
            self._unpack_observations(input_dict),
            state_batches,
            self.convert_to_tensor([1]),
        )
        action_dist = self.dist_class(dist_inputs, self.module)
        log_likelihoods = action_dist.logp(input_dict[SampleBatch.ACTIONS])
        return log_likelihoods

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        if not self.config["env_config"].get("time_aware", False):
            hit_limit = sample_batch[SampleBatch.INFOS][-1].get("TimeLimit.truncated")
            env_done = sample_batch[SampleBatch.DONES][-1]
            sample_batch[SampleBatch.DONES][-1] = False if hit_limit else env_done
        return sample_batch

    @override(Policy)
    def get_weights(self):
        buffer = io.BytesIO()
        module_state = self.module.state_dict()
        optim = () if self.optimizer is None else self.optimizer
        optims = optim if isinstance(optim, tuple) else [optim]
        torch.save([module_state] + [o.state_dict() for o in optims], buffer)
        return buffer.getvalue()

    @override(Policy)
    def set_weights(self, weights):
        buffer = io.BytesIO(weights)
        optim = () if self.optimizer is None else self.optimizer
        optims = optim if isinstance(optim, tuple) else [optim]
        states = torch.load(buffer)
        self.module.load_state_dict(states[0])
        for optim, state in zip(optims, states[1:]):
            optim.load_state_dict(state)

    def extra_action_out(self, input_dict, state_batches, module, action_dist):
        """Returns dict of extra info to include in experience batch.

        Arguments:
            input_dict (dict): Dict of model input tensors.
            state_batches (list): List of state tensors.
            model (nn.Module): Reference to the model.
            action_dist (ActionDistribution): Action dist object
                to get log-probs (e.g. for already sampled actions).
        """
        # pylint:disable=unused-argument,no-self-use
        return {}

    def convert_to_tensor(self, arr):
        """Convert an array to a PyTorch tensor in this policy's device."""
        return convert_to_tensor(arr, self.device)

    def _lazy_tensor_dict(self, sample_batch):
        tensor_batch = UsageTrackingDict(sample_batch)
        tensor_batch.set_get_interceptor(self.convert_to_tensor)
        return tensor_batch

    def _learner_stats(self, info):
        return {LEARNER_STATS_KEY: {**info, **self.get_exploration_info()}}

    @contextlib.contextmanager
    def freeze_nets(self, *names):
        """Disable gradient requirements for the desired modules in this context.

        WARNING: `.requires_grad_()` is incompatible with TorchScript.
        """
        try:
            for name in names:
                self.module[name].requires_grad_(False)
            yield
        finally:
            for name in names:
                self.module[name].requires_grad_(True)

    def __repr__(self):
        args = ["{name}(", "{observation_space}, ", "{action_space}, ", "{config}", ")"]
        config = pretty_print(self.config).rstrip("\n")
        kwargs = dict(
            name=self.__class__.__name__,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        if "\n" in config:
            config = "{\n" + config + "\n}"
            config = _addindent(config, 2)

            fmt = "\n".join(args)
            fmt = fmt.format(config=config, **kwargs)
            fmt = _addindent(fmt, 2)
        else:
            fmt = "".join(args)
            fmt = fmt.format(config=config, **kwargs)
        return fmt


def _addindent(tex_, num_spaces):
    tex = tex_.split("\n")
    # don't do anything for single-line stuff
    if len(tex) == 1:
        return tex_
    first = tex.pop(0)
    last = ""
    if len(tex) > 2:
        last = tex.pop()
    tex = [(num_spaces * " ") + line for line in tex]
    tex = "\n".join(tex)
    tex = first + "\n" + tex
    tex = tex + "\n" + last
    return tex
