# pylint:disable=missing-module-docstring
from pprint import pprint

import click


@click.command()
@click.argument("run", type=str)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=False),
    default=None,
    help="Algorithm-specific configuration for Tune variant generation "
    "(e.g. env, hyperparams). Defaults to empty dict. "
    "Custom search algorithms may ignore this. "
    "Expects a path to a python script containing a `get_config` function. ",
)
@click.option(
    "--script/--eager", "script", default=False,
)
@click.option(
    "--verbose/--quiet", default=False,
)
def main(run, config, script, verbose):
    """Print stochastic policy statistics given a config."""
    # pylint:disable=invalid-name,too-many-locals,too-many-statements
    import numpy as np
    import ray
    import torch

    import raylab
    from raylab.algorithms.registry import ALGORITHMS
    from raylab.utils.dynamic_import import import_module_from_path
    from raylab.utils.pytorch import flat_grad

    ray.init()
    raylab.register_all_environments()
    module = import_module_from_path(config)
    config = module.get_config()
    config["num_workers"] = 0
    config["module"]["torch_script"] = script

    if verbose:
        pprint(config, indent=2)
    agent_cls = ALGORITHMS[run]()
    policy = agent_cls(config).get_policy()
    config = policy.config
    obs_space, action_space = policy.observation_space, policy.action_space
    print("obs_space:", obs_space.low, obs_space.high)
    print("action_space:", action_space.low, action_space.high)
    module = policy.module

    print("=" * 80)
    print("Module:", type(module))
    print("Parameters:", sum(p.numel() for p in module.parameters()))

    print("=" * 80)
    obs = policy.convert_to_tensor([obs_space.sample()])
    act, logp = module.actor.rsample(obs)
    act.detach_()
    print("OBSERVATION:", obs)
    print("RSAMPLE:", act)
    print("LOG_PROB:", logp)
    print("MAX_ACT:", act.abs().max())

    print("=" * 80)
    logp_ = module.actor.log_prob(obs, act)
    print("EXT_LOGP:", logp_)

    print("=" * 80)
    z, log_det = module.actor.dist.transform(act, module.actor(obs), reverse=True)
    print("Z:", z)
    print("LOG_DET:", log_det)
    print("LOG_PROB_Z:", module.actor.dist.base_dist.log_prob(module.actor(obs), z))
    print("REPRODUCED:", module.actor.reproduce(obs, act))

    print("=" * 80)
    act, logp = module.actor.sample(obs, (2,))
    logp.mean().backward()
    print("SAMPLE:", act)
    print(
        "LOGP_PROB_NORM:",
        torch.cat([p.grad.reshape((-1,)) for p in module.actor.parameters()]).norm(),
    )

    print("=" * 80)
    params = list(module.actor.parameters())
    obs = policy.convert_to_tensor(np.stack([obs_space.sample() for _ in range(4000)]))
    acts, _ = module.actor.sample(obs)
    entropy = module.actor.log_prob(obs, acts).mean().neg()
    print("ENTROPY:", entropy)

    advs = torch.randn_like(obs[..., 0])
    pol_grad = flat_grad(-(module.actor.log_prob(obs, acts) * advs).mean(), params)
    print("pol_grad:", pol_grad)
    print("grad_norm(pg):", pol_grad.norm(p=2))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
