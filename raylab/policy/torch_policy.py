"""Base for all PyTorch policies."""
import textwrap
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Space
from ray.rllib import Policy
from ray.rllib import SampleBatch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.model import flatten
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import convert_to_non_torch_type
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.tune.logger import pretty_print
from torch import Tensor

from raylab.agents.options import RaylabOptions
from raylab.pytorch.utils import convert_to_tensor
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict
from raylab.utils.dictionaries import deep_merge

from .modules import get_module
from .optimizer_collection import OptimizerCollection


class TorchPolicy(Policy):
    """A Policy that uses PyTorch as a backend.

    Attributes:
        dist_class: Action distribution class for computing actions. Must be set
            by subclasses before calling `__init__`.
        device: Device in which the parameter tensors reside. All input samples
            will be converted to tensors and moved to this device
        module: The policy's neural network module. Should be compilable to
            TorchScript
        optimizers: The optimizers bound to the neural network (or submodules)
    """

    device: torch.device
    module: nn.Module
    optimizers: OptimizerCollection

    def __init__(self, observation_space: Space, action_space: Space, config: dict):
        options = self.options
        config = deep_merge(
            options.defaults,
            config,
            new_keys_allowed=True,
            whitelist=options.allow_unknown_subkeys,
            override_all_if_type_changes=options.override_all_if_type_changes,
        )

        # Allow subclasses to set `dist_class` before calling init
        action_dist = getattr(self, "dist_class", None)
        super().__init__(observation_space, action_space, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = self._make_module(observation_space, action_space, self.config)
        self.module.to(self.device)

        self.optimizers = self._make_optimizers()

        # === Policy attributes ===
        self.dist_class = action_dist
        self.dist_class.check_model_compat(self.module)
        self.framework = "torch"  # Needed to create exploration
        self.exploration = self._create_exploration()

    # ==========================================================================
    # PublicAPI
    # ==========================================================================

    @property
    def model(self):
        """The policy's NN module.

        Mostly for compatibility with RLlib's API.
        """
        return self.module

    @property
    @abstractmethod
    def options(self) -> RaylabOptions:
        """Return the options for this policy class."""

    def compile(self):
        """Optimize modules with TorchScript.

        Warnings:
            This action cannot be undone.
        """
        self.module = torch.jit.script(self.module)

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
        **kwargs,
    ):
        # pylint:disable=too-many-arguments,too-many-locals
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        input_dict = self.lazy_tensor_dict(
            {SampleBatch.CUR_OBS: obs_batch, "is_training": False}
        )
        if prev_action_batch:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        state_batches = convert_to_torch_tensor(state_batches or [], device=self.device)

        # Call the exploration before_compute_actions hook.
        self.exploration.before_compute_actions(timestep=timestep)

        dist_inputs, state_out = self._compute_module_output(
            self._unpack_observations(input_dict),
            state_batches,
            self.convert_to_tensor([1]),
        )

        # pylint:disable=not-callable
        action_dist = self.dist_class(dist_inputs, self.module)
        # pylint:enable=not-callable
        actions, logp = self.exploration.get_exploration_action(
            action_distribution=action_dist, timestep=timestep, explore=explore
        )
        input_dict[SampleBatch.ACTIONS] = actions

        # Add default and custom fetches.
        extra_fetches = self._extra_action_out(
            input_dict, state_batches, self.module, action_dist
        )

        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = logp.exp()
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        return convert_to_non_torch_type((actions, state_out, extra_fetches))

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
        input_dict = self.lazy_tensor_dict(
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
        # pylint:disable=not-callable
        action_dist = self.dist_class(dist_inputs, self.module)
        # pylint:enable=not-callable
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
    def get_weights(self) -> dict:
        return {
            "module": convert_to_non_torch_type(self.module.state_dict()),
            # Optimizer state dicts don't store tensors, only ids
            "optimizers": self.optimizers.state_dict(),
        }

    @override(Policy)
    def set_weights(self, weights: dict):
        self.module.load_state_dict(
            convert_to_torch_tensor(weights["module"], device=self.device)
        )
        # Optimizer state dicts don't store tensors, only ids
        self.optimizers.load_state_dict(weights["optimizers"])

    def convert_to_tensor(self, arr) -> Tensor:
        """Convert an array to a PyTorch tensor in this policy's device.

        Args:
            arr (array_like): object which can be converted using `np.asarray`
        """
        return convert_to_tensor(arr, self.device)

    def lazy_tensor_dict(self, sample_batch: SampleBatch) -> UsageTrackingDict:
        """Convert a sample batch into a dictionary of lazy tensors.

        The sample batch is wrapped with a UsageTrackingDict to convert array-
        likes into tensors upon querying.

        Args:
            sample_batch: the sample batch to convert

        Returns:
            A dictionary which intercepts key queries to lazily convert arrays
            to tensors.
        """
        tensor_batch = UsageTrackingDict(sample_batch)
        tensor_batch.set_get_interceptor(self.convert_to_tensor)
        return tensor_batch

    def __repr__(self):
        name = self.__class__.__name__
        args = [f"{self.observation_space},", f"{self.action_space},"]

        config = pretty_print(self.config).rstrip("\n")
        if "\n" in config:
            config = textwrap.indent(config, " " * 2)
            config = "{\n" + config + "\n}"

            args += [config]
            args_repr = "\n".join(args)
            args_repr = textwrap.indent(args_repr, " " * 2)
            constructor = f"{name}(\n{args_repr}\n)"
        else:
            args += [config]
            args_repr = " ".join(args[1:-1])
            constructor = f"{name}({args_repr})"
        return constructor

    # ==========================================================================
    # InternalAPI
    # ==========================================================================

    @staticmethod
    def _make_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
        """Build the PyTorch nn.Module to be used by this policy.

        Args:
            obs_space: the observation space for this policy
            action_space: the action_space for this policy
            config: the user config containing the 'module' key

        Returns:
            A neural network module.
        """
        return get_module(obs_space, action_space, config["module"])

    def _make_optimizers(self) -> OptimizerCollection:
        """Build PyTorch optimizers to use.

        The result will be set as the policy's optimizer collection.

        The user should update the optimizer collection (mutable mapping)
        returned by the base implementation.

        Returns:
            A mapping from names to optimizer instances
        """
        # pylint:disable=no-self-use
        return OptimizerCollection()

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

    def _compute_module_output(
        self, input_dict: TensorDict, state: List[Tensor], seq_lens: Tensor
    ) -> Tuple[TensorDict, List[Tensor]]:
        """Call the module with the given input tensors and state.

        This mirrors the method used by RLlib to execute the forward pass. Nested
        observation tensors are unpacked before this function is called.

        Subclasses should override this for custom forward passes (e.g., for recurrent
        networks).

        Args:
            input_dict: dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training"
            state: list of state tensors with sizes matching those returned
                by get_initial_state + the batch dimension
            seq_lens: 1d tensor holding input sequence lengths

        Returns:
            A tuple containg an input dictionary to the policy's `dist_class`
            and a list of rnn state tensors
        """
        # pylint:disable=unused-argument,no-self-use
        return {"obs": input_dict["obs"]}, state

    def _extra_action_out(
        self,
        input_dict: TensorDict,
        state_batches: List[Tensor],
        module: nn.Module,
        action_dist: ActionDistribution,
    ) -> StatDict:
        """Returns dict of extra info to include in experience batch.

        Args:
            input_dict: Dict of model input tensors.
            state_batches: List of state tensors.
            model: Reference to the model.
            action_dist: Action dist object
                to get log-probs (e.g. for already sampled actions).
        """
        # pylint:disable=unused-argument,no-self-use
        return {}

    # ==========================================================================
    # Unimplemented Policy methods
    # ==========================================================================

    def export_model(self, export_dir):
        pass

    def export_checkpoint(self, export_dir):
        pass

    def import_model_from_h5(self, import_file):
        pass


def _to_numpy_state_dict(mapping):
    for key, val in mapping.items():
        if torch.is_tensor(val):
            mapping[key] = val.cpu().detach().numpy()
        elif isinstance(val, dict):
            _to_numpy_state_dict(val)


def _from_numpy_state_dict(mapping, device=None):
    for key, val in mapping.items():
        if isinstance(val, np.ndarray):
            mapping[key] = convert_to_tensor(val, device)
        elif isinstance(val, dict):
            _from_numpy_state_dict(val, device)
