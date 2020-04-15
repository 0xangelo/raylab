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
@click.option(
    "--tensorboard-dir",
    type=click.Path(exists=None, file_okay=False, dir_okay=True),
    default="runs/",
)
def main(run, config, script, verbose, tensorboard_dir):
    """Print stochastic policy statistics given a config."""
    # pylint:disable=import-outside-toplevel,too-many-locals
    import os
    import shutil

    import ray
    from torch.utils.tensorboard import SummaryWriter
    import raylab
    from raylab.algorithms.registry import ALGORITHMS
    from raylab.utils.dynamic_import import import_module_from_path

    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)

    ray.init()
    raylab.register_all_environments()
    module = import_module_from_path(config)
    config = module.get_config()
    config["num_workers"] = 0
    config["module"]["torch_script"] = script

    agent_cls = ALGORITHMS[run]()
    policy = agent_cls(config).get_policy()
    obs_space, action_space = policy.observation_space, policy.action_space

    print("============================= TEST INIT ===================================")
    test_init(policy, obs_space, action_space, verbose=verbose)
    print("============================= TEST SAMPLER ================================")
    test_sampler(policy, obs_space)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("============================= TEST BATCH ==================================")
    test_big_batch(policy, obs_space, writer)
    # print("============================= PLOT GRAPH ================================")
    # if not script:
    #     tensorboard_graph(policy, obs_space, writer)
    writer.close()


def tensorboard_graph(policy, obs_space, writer):
    # pylint:disable=missing-docstring,import-outside-toplevel,arguments-differ
    # pylint:disable=c-extension-no-member,protected-access,too-many-locals
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard._pytorch_graph import (
        RunMetadata,
        StepStats,
        DeviceStepStats,
        GraphDef,
        VersionDef,
        parse,
    )

    class FullActor(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            act, logp = self.actor.sample(obs)
            return act, logp

    args = policy.convert_to_tensor([obs_space.sample()]).detach()
    model = FullActor(policy.module.actor)
    trace = torch.jit.trace(model, args, check_trace=False)
    with torch.onnx.set_training(model, False):
        print(trace)
        graph = trace.graph
        torch._C._jit_pass_inline(graph)
    list_of_nodes = parse(graph, trace, args)
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    graphdef = GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
    writer._get_file_writer().add_graph(graphdef)


def test_init(policy, obs_space, action_space, verbose=False):
    # pylint:disable=missing-function-docstring
    config = policy.config
    if verbose:
        pprint(config, indent=2)
    print("obs_space:", obs_space.low, obs_space.high)
    print("action_space:", action_space.low, action_space.high)
    module = policy.module

    print("Module:", type(module))
    print("Parameters:", sum(p.numel() for p in module.parameters()))


def test_sampler(policy, obs_space):
    # pylint:disable=import-outside-toplevel,no-member,missing-function-docstring
    # pylint:disable=invalid-name
    import torch

    module = policy.module

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
    print("LOG_PROB_Z:", module.actor.dist.base_dist.log_prob(z, module.actor(obs)))
    print("REPRODUCED:", module.actor.reproduce(obs, act))

    print("=" * 80)
    act, logp = module.actor.sample(obs, (2,))
    logp.mean().backward()
    print("SAMPLE:", act)
    print("SAMPLE_LOGP:", logp)
    print("BATCH_EXT_LOGP:", module.actor.log_prob(obs, act))
    print(
        "LOGP_PROB_NORM:",
        torch.cat([p.grad.reshape((-1,)) for p in module.actor.parameters()]).norm(),
    )


def test_big_batch(policy, obs_space, writer):
    # pylint:disable=import-outside-toplevel,missing-function-docstring,no-member
    # pylint:disable=too-many-locals
    import numpy as np
    import torch
    from raylab.utils.pytorch import flat_grad

    module = policy.module
    params = list(module.actor.parameters())

    obs = policy.convert_to_tensor(np.stack([obs_space.sample() for _ in range(4000)]))
    acts, logp = module.actor.sample(obs)
    for idx in range(acts.size(-1)):
        writer.add_histogram(f"Actions/{idx}", acts[..., idx], 0)
    writer.add_histogram("Log Probs", logp, 0)

    print("BATCHED_SAMPLES:", acts.shape)
    print("                ", acts.abs().max())
    print("BATCHED_LOGP:", logp)
    isnan = torch.isnan(logp)
    print("NAN_LOGPs:", logp[isnan], isnan.sum())

    logp_ex = module.actor.log_prob(obs, acts)
    isnan = torch.isnan(logp_ex)
    print("BATCH_EXT_LOGP:", logp_ex)
    print("               ", logp_ex.shape)
    print("               ", isnan.sum())

    print("MAX(BATCH_LOPG - BATCH_EXT_LOGP):", (logp - logp_ex).max())

    entropy = -logp_ex.mean()
    print("ENTROPY:", entropy)

    advs = torch.randn_like(obs[..., 0])
    pol_grad = flat_grad(-(module.actor.log_prob(obs, acts) * advs).mean(), params)
    print("pol_grad:", pol_grad)
    print("grad_norm(pg):", pol_grad.norm(p=2))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
