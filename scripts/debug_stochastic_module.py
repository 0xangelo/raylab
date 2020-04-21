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
@click.option(
    "--component",
    type=click.Choice(["actor", "model"]),
    default="actor",
    show_default=True,
)
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
    test_sampler(agent.get_policy(), args)
    writer = SummaryWriter(log_dir=args["tensorboard_dir"])
    rollout = produce_rollout(agent)
    test_rollout(agent.get_policy(), rollout, writer, args)

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


def test_sampler(policy, args):
    # pylint:disable=no-member,invalid-name
    print(" TEST SAMPLER ".center(80, "="))
    import torch

    obs_space, action_space = policy.observation_space, policy.action_space

    if args["component"] == "actor":
        module = policy.module.actor
        inputs = (policy.convert_to_tensor([obs_space.sample()]),)
    elif args["component"] == "model":
        module = policy.module.model
        inputs = (
            policy.convert_to_tensor([obs_space.sample()]),
            policy.convert_to_tensor([action_space.sample()]),
        )
    else:
        raise ValueError

    rsamp, logp = module.rsample(*inputs)
    rsamp.detach_()
    print("INPUTS".ljust(12), "|", *inputs)
    print("RSAMPLE".ljust(12), "|", rsamp)
    print("RSAMPLE_LOGP".ljust(12), "|", logp)
    print("MAX_SAMPLE".ljust(12), "|", rsamp.abs().max())

    print("-" * 60)
    logp_ = module.log_prob(*inputs, rsamp)
    print("EXT_LOGP".ljust(12), "|", logp_)

    print("-" * 60)
    z, log_det = module.dist.transform(rsamp, module(*inputs), reverse=True)
    print("Z".ljust(12), "|", z)
    print("LOG_DET".ljust(12), "|", log_det)
    print(
        "LOGP_Z".ljust(12), "|", module.dist.base_dist.log_prob(z, module(*inputs)),
    )
    samp_, logp_ = module.reproduce(*inputs, rsamp)
    print("REPR_SAMP".ljust(12), "|", samp_)
    print("REPR_LOGP".ljust(12), "|", logp_)

    print("-" * 60)
    sample_shape = (2,)
    print("SAMP_SHAPE".ljust(12), "|", sample_shape)
    samp, logp = module.sample(*inputs, sample_shape)
    logp.mean().backward()
    print("SAMPLE".ljust(12), "|", samp)
    print("SAMPLE_LOGP".ljust(12), "|", logp)
    print("EXT_LOGP".ljust(12), "|", module.log_prob(*inputs, samp))
    print(
        "EXT_NORM".ljust(12),
        "|",
        torch.cat([p.grad.flatten() for p in module.parameters()]).norm(),
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


def test_rollout(policy, rollout, writer, args):
    # pylint:disable=no-member
    from ray.rllib.policy.policy import ACTION_LOGP
    from ray.rllib.policy.sample_batch import SampleBatch
    from raylab.utils.pytorch import flat_grad

    print(" TEST ROLLOUT ".center(80, "="))
    if args["component"] == "actor":
        module = policy.module.actor
        inputs = (rollout[SampleBatch.CUR_OBS],)
        samps, logp = rollout[SampleBatch.ACTIONS], rollout[ACTION_LOGP]
    elif args["component"] == "model":
        module = policy.module.model
        inputs = (
            rollout[SampleBatch.CUR_OBS],
            rollout[SampleBatch.ACTIONS],
        )
        samps, logp = policy.module.model.sample(
            rollout[SampleBatch.CUR_OBS], rollout[SampleBatch.ACTIONS],
        )
    else:
        raise ValueError

    test_samples(module, inputs, samps, logp, args["component"], writer)

    loss = -module.log_prob(*inputs, samps.detach()).mean()
    nll_grad = flat_grad(loss, module.parameters())
    print("NLL grad:", nll_grad)
    print("grad_norm(nll):", nll_grad.norm(p=2))


def test_samples(module, inputs, samps, logp, label, writer):
    # pylint:disable=no-member,too-many-arguments
    import torch

    for idx in range(samps.size(-1)):
        writer.add_histogram(f"{label}/{idx}", samps[..., idx], 0)
    writer.add_histogram(f"{label}_log_probs", logp, 0)

    idxs = torch.randperm(samps.size(0))[:10]
    print(" SAMPLES ".center(60, "-"))
    print("Samps".ljust(16), samps[idxs])
    print("Shape".ljust(16), samps.shape)
    print("Abs Max".ljust(16), samps.abs().max())
    print("Mean".ljust(16), samps.mean())
    print("Logp".ljust(16), logp[idxs])
    isnan = torch.isnan(logp)
    print("NaNs".ljust(16), isnan.sum())

    logp_ex = module.log_prob(*inputs, samps)
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
