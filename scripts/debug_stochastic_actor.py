# pylint:disable=missing-module-docstring,missing-function-docstring,import-outside-toplevel
from pprint import pprint
import warnings

import click

from raylab.cli.utils import initialize_raylab


@click.command()
@click.argument("agent_name", type=str)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=False),
    default=None,
)
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
@click.option("--eval/--no-eval", default=False)
@initialize_raylab
def main(**args):
    """Print stochastic policy statistics given a config."""
    import ray
    from torch.utils.tensorboard import SummaryWriter

    ray.init()
    prepare_tensorboard_dir(args["tensorboard_dir"])
    agent = get_agent(
        args["agent_name"],
        args["config"],
        args["checkpoint"],
        args["eval"],
        args["script"],
    )

    test_init(agent.get_policy(), verbose=args["verbose"])
    test_sampler(agent.get_policy())
    writer = SummaryWriter(log_dir=args["tensorboard_dir"])
    rollout = produce_rollout(agent)
    test_rollout(agent.get_policy(), rollout, writer)

    # if not script:
    #     tensorboard_graph(policy, obs_space, writer)
    writer.close()


def get_agent(agent_name, config_path, checkpoint, evaluate, script):
    from ray.rllib.utils import merge_dicts
    from raylab.utils.checkpoints import get_config_from_checkpoint, get_agent_cls
    from raylab.utils.dynamic_import import import_module_from_path

    msg = "Either config or checkpoint can be chosen."
    assert (config_path is None) != (checkpoint is None), msg

    if config_path is not None:
        config = import_module_from_path(config_path).get_config()
        if evaluate:
            if "evaluation_config" not in config:
                warnings.warn("Evaluation agent requested but none in config.")
            else:
                eval_conf = config["evaluation_config"]
                config = merge_dicts(config, eval_conf)
    else:

        config = get_config_from_checkpoint(checkpoint, evaluate)

    config["num_workers"] = 0
    config["module"]["torch_script"] = script
    agent_cls = get_agent_cls(agent_name)
    agent = agent_cls(config)
    if checkpoint:
        agent.restore(checkpoint)
    return agent


def prepare_tensorboard_dir(tensorboard_dir):
    import os
    import shutil

    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)


def test_init(policy, verbose=False):
    print(" TEST INIT ".center(80, "="))
    config = policy.config
    if verbose:
        pprint(config, indent=2)
    obs_space, action_space = policy.observation_space, policy.action_space
    module = policy.module

    print("obs_space".ljust(12), "|", obs_space.low, obs_space.high)
    print("action_space".ljust(12), "|", action_space.low, action_space.high)
    print("Module".ljust(12), "|", type(module))
    print("Parameters".ljust(12), "|", sum(p.numel() for p in module.parameters()))


def test_sampler(policy):
    # pylint:disable=no-member,invalid-name
    print(" TEST SAMPLER ".center(80, "="))
    import torch

    module = policy.module
    obs_space = policy.observation_space

    obs = policy.convert_to_tensor([obs_space.sample()])
    act, logp = module.actor.rsample(obs)
    act.detach_()
    print("OBSERVATION".ljust(12), "|", obs)
    print("RSAMPLE".ljust(12), "|", act)
    print("RSAMPLE_LOGP".ljust(12), "|", logp)
    print("MAX_ACT".ljust(12), "|", act.abs().max())

    print("-" * 60)
    logp_ = module.actor.log_prob(obs, act)
    print("EXT_LOGP".ljust(12), "|", logp_)

    print("-" * 60)
    z, log_det = module.actor.dist.transform(act, module.actor(obs), reverse=True)
    print("Z".ljust(12), "|", z)
    print("LOG_DET".ljust(12), "|", log_det)
    print(
        "LOGP_Z".ljust(12),
        "|",
        module.actor.dist.base_dist.log_prob(z, module.actor(obs)),
    )
    act_, logp_ = module.actor.reproduce(obs, act)
    print("REPR_ACT".ljust(12), "|", act_)
    print("REPR_LOGP".ljust(12), "|", logp_)

    print("-" * 60)
    sample_shape = (2,)
    print("SAMP_SHAPE".ljust(12), "|", sample_shape)
    act, logp = module.actor.sample(obs, sample_shape)
    logp.mean().backward()
    print("SAMPLE".ljust(12), "|", act)
    print("SAMPLE_LOGP".ljust(12), "|", logp)
    print("EXT_LOGP".ljust(12), "|", module.actor.log_prob(obs, act))
    print(
        "EXT_NORM".ljust(12),
        "|",
        torch.cat([p.grad.reshape((-1,)) for p in module.actor.parameters()]).norm(),
    )


def produce_rollout(agent):
    # pylint:disable=protected-access
    print(" PRODUCE ROLLOUT ".center(80, "="))
    import logging

    logging.getLogger("ray.rllib").setLevel(logging.DEBUG)
    worker = agent.workers.local_worker()
    worker.batch_mode = "complete_episodes"
    worker.rollout_fragment_length = 4000

    print("global_vars:", agent.global_vars)
    print("policy_timestep:", agent.get_policy().global_timestep)
    print("worker_vars:", worker.global_vars)
    episode = worker.sample()
    return agent.get_policy()._lazy_tensor_dict(episode)


def test_rollout(policy, rollout, writer):
    # pylint:disable=no-member,too-many-locals
    print(" TEST ROLLOUT ".center(80, "="))
    import torch
    from ray.rllib.policy.policy import ACTION_LOGP
    from ray.rllib.policy.sample_batch import SampleBatch
    from raylab.utils.pytorch import flat_grad

    module = policy.module
    params = list(module.actor.parameters())

    obs = rollout[SampleBatch.CUR_OBS]
    acts, logp = rollout[SampleBatch.ACTIONS], rollout[ACTION_LOGP]
    for idx in range(acts.size(-1)):
        writer.add_histogram(f"Actions/{idx}", acts[..., idx], 0)
    writer.add_histogram("Log Probs", logp, 0)

    idxs = torch.randperm(obs.size(0))[:10]
    print(" SAMPLES ".center(60, "-"))
    print("Acts".ljust(16), acts[idxs])
    print("Shape".ljust(16), acts.shape)
    print("Abs Max".ljust(16), acts.abs().max())
    print("Mean".ljust(16), acts.mean())
    print("Logp".ljust(16), logp[idxs])
    isnan = torch.isnan(logp)
    print("NaNs".ljust(16), isnan.sum())

    logp_ex = module.actor.log_prob(obs, acts)
    isnan = torch.isnan(logp_ex)
    print(" EXT_LOGP ".center(60, "-"))
    print(logp_ex[idxs])
    print("Shape".ljust(16), logp_ex.shape)
    print("NaNs".ljust(16), isnan.sum())

    print()
    print("MAX(BATCH_LOPG - BATCH_EXT_LOGP): ", (logp - logp_ex).max())
    print("MEAN(BATCH_LOPG - BATCH_EXT_LOGP):", (logp - logp_ex).mean())

    entropy = -logp_ex.mean()
    print()
    print("ENTROPY:", entropy)

    acts = rollout[SampleBatch.ACTIONS]
    advs = torch.randn_like(obs[..., 0])
    pol_grad = flat_grad(-(module.actor.log_prob(obs, acts) * advs).mean(), params)
    print("pol_grad:", pol_grad)
    print("grad_norm(pg):", pol_grad.norm(p=2))


def tensorboard_graph(policy, obs_space, writer):
    # pylint:disable=arguments-differ,too-many-locals
    # pylint:disable=c-extension-no-member,protected-access
    print(" PLOT GRAPH ".center(80, "="))
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
        # pylint:disable=missing-class-docstring
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


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
