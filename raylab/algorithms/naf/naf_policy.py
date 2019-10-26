"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
import torch.distributions as dists
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.modules as mods
import raylab.utils.pytorch as torch_util
import raylab.algorithms.naf.naf_module as nafm
import raylab.policy as raypi


class NAFTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_naf(obs_space, action_space, config))

        def make_value(obs_space, config):
            logits = self._make_encoder(obs_space, config)
            return nn.Sequential(logits, mods.ValueFunction(logits.out_features))

        module.target_value = nn.ModuleList(
            [make_value(obs_space, config) for _ in module.value]
        )
        module.target_value.load_state_dict(module.value.state_dict())

        module.policy = nn.Sequential(module.logits, module.mu, module.squash)
        # Configure sampler module based on exploration strategy
        if config["exploration"] == "full_gaussian":
            params_module = nafm.MultivariateNormalParams(
                module.logits, module.mu, module.tril
            )
            rsample_module = mods.DistRSample(
                dists.MultivariateNormal,
                low=self.convert_to_tensor(action_space.low),
                high=self.convert_to_tensor(action_space.high),
            )
            module.sampler = nn.Sequential(params_module, rsample_module)
        elif config["exploration"] == "diag_gaussian":
            expl_noise = mods.GaussianNoise(config["diag_gaussian_stddev"])
            module.sampler = nn.Sequential(
                module.logits, module.mu, expl_noise, module.squash
            )
        elif config["exploration"] == "parameter_noise":
            logits_module = self._make_encoder(obs_space, config)
            module.perturbed_policy = module.sampler = nn.Sequential(
                logits_module,
                mods.NormalizedLinear(
                    in_features=logits_module.out_features,
                    out_features=action_space.shape[0],
                    beta=config["beta"],
                ),
                mods.TanhSquash(
                    self.convert_to_tensor(action_space.low),
                    self.convert_to_tensor(action_space.high),
                ),
            )
        else:
            module.sampler = module.policy

        return module

    def _make_naf(self, obs_space, action_space, config):
        modules = self._make_components(obs_space, action_space, config)
        naf_module = nafm.NAF(
            modules["logits"],
            modules["value"],
            modules["tril"],
            nn.Sequential(modules["mu"], modules["squash"]),
        )
        modules["naf"] = nn.ModuleList([naf_module])
        modules["value"] = nn.ModuleList(
            [nn.Sequential(modules["logits"], modules["value"])]
        )
        if config["clipped_double_q"]:
            twin_modules = self._make_components(obs_space, action_space, config)
            twin_naf = nafm.NAF(
                twin_modules["logits"],
                twin_modules["value"],
                twin_modules["tril"],
                nn.Sequential(twin_modules["mu"], twin_modules["squash"]),
            )
            modules["naf"].append(twin_naf)
            modules["value"].append(
                nn.Sequential(twin_modules["logits"], twin_modules["value"])
            )

        return modules

    def _make_components(self, obs_space, action_space, config):
        logits = self._make_encoder(obs_space, config)
        return {
            "logits": logits,
            "value": mods.ValueFunction(logits.out_features),
            "mu": mods.NormalizedLinear(
                in_features=logits.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            ),
            "squash": mods.TanhSquash(
                self.convert_to_tensor(action_space.low),
                self.convert_to_tensor(action_space.high),
            ),
            "tril": mods.TrilMatrix(logits.out_features, action_space.shape[0]),
        }

    @staticmethod
    def _make_encoder(obs_space, config):
        return mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=config["module"]["units"],
            activation=config["module"]["activation"],
            layer_norm=config["module"].get(
                "layer_norm", config["exploration"] == "parameter_noise"
            ),
            **config["module"]["initializer_options"],
        )

    @override(raypi.TorchPolicy)
    def optimizer(self):
        cls = torch_util.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        return cls(self.module.naf.parameters(), **options)

    @torch.no_grad()
    @override(raypi.TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

        if self.config["greedy"]:
            actions = self.module.policy(obs_batch)
        elif self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["exploration"] == "full_gaussian":
            actions = self._multivariate_gaussian_actions(obs_batch)
        else:
            actions = self.module.sampler(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    # === Action Sampling ===
    def _multivariate_gaussian_actions(self, obs_batch):
        loc, scale_tril = self.module.sampler(obs_batch)
        scale_coeff = self.config["scale_tril_coeff"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=scale_tril * scale_coeff
        )
        actions = dist.sample()
        return actions

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        loss, info = self.compute_loss(batch_tensors, self.module, self.config)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info())
        self._optimizer.step()

        self.update_targets("value", "target_value")
        return self._learner_stats(info)

    def compute_loss(self, batch_tensors, module, config):
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
            target_values = self._compute_critic_targets(batch_tensors, module, config)

        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]
        action_values = torch.cat([m(obs, actions) for m in module.naf], dim=-1)
        loss_fn = torch.nn.MSELoss()
        td_error = loss_fn(
            action_values, target_values.unsqueeze(-1).expand_as(action_values)
        )

        stats = {
            "q_mean": action_values.mean().item(),
            "q_max": action_values.max().item(),
            "q_min": action_values.min().item(),
            "td_error": td_error.item(),
        }
        return td_error, stats

    @staticmethod
    def _compute_critic_targets(batch_tensors, module, config):
        rewards = batch_tensors[SampleBatch.REWARDS]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_vals, _ = torch.cat(
            [m(next_obs) for m in module.target_value], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.naf.parameters(), float("inf")
            )
        }
