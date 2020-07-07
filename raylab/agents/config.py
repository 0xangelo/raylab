"""Utilities for default configurations and info messages."""
from textwrap import dedent
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import tree
from dataclasses_json.core import Json
from ray.rllib.agents.trainer import with_base_config

COMMON_INFO = {
    # === Settings for Rollout Worker processes ===
    "num_workers": """\
    Number of rollout worker actors to create for parallel sampling. Setting
    this to 0 will force rollouts to be done in the trainer actor.
    """,
    "num_envs_per_worker": """\
    Number of environments to evaluate vectorwise per worker. This enables
    model inference batching, which can improve performance for inference
    bottlenecked workloads.
    """,
    "rollout_fragment_length": """\
    Divide episodes into fragments of this many steps each during rollouts.
    Sample batches of this size are collected from rollout workers and
    combined into a larger batch of `train_batch_size` for learning.

    For example, given rollout_fragment_length=100 and train_batch_size=1000:
      1. RLlib collects 10 fragments of 100 steps each from rollout workers.
      2. These fragments are concatenated and we perform an epoch of SGD.

    When using multiple envs per worker, the fragment size is multiplied by
    `num_envs_per_worker`. This is since we are collecting steps from
    multiple envs in parallel. For example, if num_envs_per_worker=5, then
    rollout workers will return experiences in chunks of 5*100 = 500 steps.

    The dataflow here can vary per algorithm. For example, PPO further
    divides the train batch into minibatches for multi-epoch SGD.
    """,
    "sample_batch_size": """\
    Deprecated; renamed to `rollout_fragment_length` in 0.8.4.
    """,
    "batch_mode": """\
    Whether to rollout "complete_episodes" or "truncate_episodes" to
    `rollout_fragment_length` length unrolls. Episode truncation guarantees
    evenly sized batches, but increases variance as the reward-to-go will
    need to be estimated at truncation boundaries.
    """,
    # === Settings for the Trainer process ===
    "num_gpus": """\
    Number of GPUs to allocate to the trainer process. Note that not all
    algorithms can take advantage of trainer GPUs. This can be fractional
    (e.g., 0.3 GPUs).
    """,
    "train_batch_size": """\
    Training batch size, if applicable. Should be >= rollout_fragment_length.
    Samples batches will be concatenated together to a batch of this size,
    which is then passed to SGD.
    """,
    "model": """\
    Arguments to pass to the policy model. See models/catalog.py for a full
    list of the available model options.
    """,
    "optimizer": """\
    Arguments to pass to the policy optimizer. These vary by optimizer.
    """,
    # === Environment Settings ===
    "gamma": """\
    Discount factor of the MDP.
    """,
    "horizon": """\
    Number of steps after which the episode is forced to terminate. Defaults
    to `env.spec.max_episode_steps` (if present) for Gym envs.
    """,
    "soft_horizon": """\
    Calculate rewards but don't reset the environment when the horizon is
    hit. This allows value estimation and RNN state to span across logical
    episodes denoted by horizon. This only has an effect if horizon != inf.
    """,
    "no_done_at_end": """\
    Don't set 'done' at the end of the episode. Note that you still need to
    set this if soft_horizon=True, unless your env is actually running
    forever without returning done=True.
    """,
    "env_config": """\
    Arguments to pass to the env creator.
    """,
    "env": """\
    Environment name can also be passed via config.
    """,
    "normalize_actions": """\
    Unsquash actions to the upper and lower bounds of env's action space
    """,
    "clip_rewards": """\
    Whether to clip rewards prior to experience postprocessing. Setting to
    None means clip for Atari only.
    """,
    "clip_actions": """\
    Whether to np.clip() actions to the action space low/high range spec.
    """,
    "preprocessor_pref": """\
    Whether to use rllib or deepmind preprocessors by default
    """,
    "lr": """\
    The default learning rate. Not used by Raylab
    """,
    # === Debug Settings ===
    "monitor": """\
    Whether to write episode stats and videos to the agent log dir. This is
    typically located in ~/ray_results.
    """,
    "log_level": """\
    Set the ray.rllib.* log level for the agent process and its workers.
    Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    periodically print out summaries of relevant internal dataflow (this is
    also printed out once at startup at the INFO level). When using the
    `rllib train` command, you can also use the `-v` and `-vv` flags as
    shorthand for INFO and DEBUG.
    """,
    "callbacks": """\
    Callbacks that will be run during various phases of training. See the
    `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
    for more usage information.
    """,
    "ignore_worker_failures": """\
    Whether to attempt to continue training if a worker crashes. The number
    of currently healthy workers is reported as the "num_healthy_workers"
    metric.
    """,
    "log_sys_usage": """\
    Log system resource metrics to results. This requires `psutil` to be
    installed for sys stats, and `gputil` for GPU metrics.
    """,
    "fake_sampler": """\
    Use fake (infinite speed) sampler. For testing only.
    """,
    # === Deep Learning Framework Settings ===
    "framework": """\
    tf: TensorFlow
    tfe: TensorFlow eager
    torch: PyTorch
    """,
    "eager_tracing": """\
    Enable tracing in eager mode. This greatly improves performance, but
    makes it slightly harder to debug since Python code won't be evaluated
    after the initial eager pass. Only possible if framework=tfe.
    """,
    "no_eager_on_workers": """\
    Disable eager execution on workers (but allow it on the driver). This
    only has an effect if eager is enabled.
    """,
    # === Exploration Settings ===
    "explore": """\
    Default exploration behavior, iff `explore`=None is passed into
    compute_action(s).
    Set to False for no exploration behavior (e.g., for evaluation).
    """,
    "exploration_config": {
        "__help__": """\
        Provide a dict specifying the Exploration object's config.
        """,
        "type": """\
        The Exploration class to use. In the simplest case, this is the name
        (str) of any class present in the `rllib.utils.exploration` package.
        You can also provide the python class directly or the full location
        of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        EpsilonGreedy").
        """,
    },
    # === Evaluation Settings ===
    "evaluation_interval": """\
    Evaluate with every `evaluation_interval` training iterations.
    The evaluation stats will be reported under the "evaluation" metric key.
    Note that evaluation is currently not parallelized, and that for Ape-X
    metrics are already only reported for the lowest epsilon workers.
    """,
    "evaluation_num_episodes": """\
    Number of episodes to run per evaluation period. If using multiple
    evaluation workers, we will run at least this many episodes total.
    """,
    "in_evaluation": """\
    Internal flag that is set to True for evaluation workers.
    """,
    "evaluation_config": """\
    Typical usage is to pass extra args to evaluation env creator
    and to disable exploration by computing deterministic actions.
    IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    policy, even if this is a stochastic one. Setting "explore=False" here
    will result in the evaluation workers not using this optimal policy!

    Example: overriding env_config, exploration, etc:
    "env_config": {...},
    "explore": False
    """,
    "evaluation_num_workers": """\
    Number of parallel workers to use for evaluation. Note that this is set
    to zero by default, which means evaluation will be run in the trainer
    process. If you increase this, it will increase the Ray resource usage
    of the trainer since evaluation workers are created separately from
    rollout workers.
    """,
    "custom_eval_function": """\
    Customize the evaluation method. This must be a function of signature
    (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    Trainer._evaluate() method to see the default implementation. The
    trainer guarantees all eval workers have the latest policy state before
    this function is called.
    """,
    # === Advanced Rollout Settings ===
    "sample_async": """\
    Use a background thread for sampling (slightly off-policy, usually not
    advisable to turn on unless your env specifically requires it).
    """,
    "observation_filter": """\
    Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    """,
    "synchronize_filters": """\
    Whether to synchronize the statistics of remote filters.
    """,
    "tf_session_args": {
        "__help__": """\
        Configures TF for single-process operation by default.
        """,
        "intra_op_parallelism_threads": """\
        note: overriden by `local_tf_session_args`
        """,
        "inter_op_parallelism_threads": "",
        "gpu_options": {"allow_growth": ""},
        "log_device_placement": "",
        "device_count": {"CPU": ""},
        "allow_soft_placement": """\
        required by PPO multi-gpu
        """,
    },
    "local_tf_session_args": {
        "__help__": """\
        Override the following tf session args on the local worker

        Allow a higher level of parallelism by default, but not unlimited
        since that can cause crashes with many concurrent drivers.
        """,
        "intra_op_parallelism_threads": "",
        "inter_op_parallelism_threads": "",
    },
    "compress_observations": """\
    Whether to LZ4 compress individual observations
    """,
    "collect_metrics_timeout": """\
    Wait for metric batches for at most this many seconds. Those that
    have not returned in time will be collected in the next train iteration.
    """,
    "metrics_smoothing_episodes": """\
    Smooth metrics over this many episodes.
    """,
    "remote_worker_envs": """\
    If using num_envs_per_worker > 1, whether to create those new envs in
    remote processes instead of in the same worker. This adds overheads, but
    can make sense if your envs can take much time to step / reset
    (e.g., for StarCraft). Use this cautiously; overheads are significant.
    """,
    "remote_env_batch_wait_ms": """\
    Timeout that remote workers are waiting when polling environments.
    0 (continue when at least one env is ready) is a reasonable default,
    but optimal value could be obtained by measuring your environment
    step / reset and model inference perf.
    """,
    "min_iter_time_s": """\
    Minimum time per train iteration (frequency of metrics reporting).
    """,
    "timesteps_per_iteration": """\
    Minimum env steps to optimize for per train call. This value does
    not affect learning, only the length of train iterations.
    """,
    "seed": """\
    This argument, in conjunction with worker_index, sets the random seed of
    each worker, so that identically configured trials will have identical
    results. This makes experiments reproducible.
    """,
    "extra_python_environs_for_driver": """\
    Any extra python env vars to set in the trainer process, e.g.,
    {"OMP_NUM_THREADS": "16"}
    """,
    "extra_python_environs_for_worker": """\
    The extra python environments need to set for worker processes.
    """,
    # === Advanced Resource Settings ===
    "num_cpus_per_worker": """\
    Number of CPUs to allocate per worker.
    """,
    "num_gpus_per_worker": """\
    Number of GPUs to allocate per worker. This can be fractional. This is
    usually needed only if your env itself requires a GPU (i.e., it is a
    GPU-intensive video game), or model inference is unusually expensive.
    """,
    "custom_resources_per_worker": """\
    Any custom Ray resources to allocate per worker.
    """,
    "num_cpus_for_driver": """\
    Number of CPUs to allocate for the trainer. Note: this only takes effect
    when running in Tune. Otherwise, the trainer runs in the main program.
    """,
    "memory": """\
    You can set these memory quotas to tell Ray to reserve memory for your
    training run. This guarantees predictable execution, but the tradeoff is
    if your workload exceeeds the memory quota it will fail.
    Heap memory to reserve for the trainer process (0 for unlimited). This
    can be large if your are using large train batches, replay buffers, etc.
    """,
    "object_store_memory": """\
    Object store memory to reserve for the trainer process. Being large
    enough to fit a few copies of the model weights should be sufficient.
    This is enabled by default since models are typically quite small.
    """,
    "memory_per_worker": """\
    Heap memory to reserve for each worker. Should generally be small unless
    your environment is very heavyweight.
    """,
    "object_store_memory_per_worker": """\
    Object store memory to reserve for each worker. This only needs to be
    large enough to fit a few sample batches at a time. This is enabled
    by default since it almost never needs to be larger than ~200MB.
    """,
    # === Offline Datasets ===
    "input": """\
    Specify how to generate experiences:
     - "sampler": generate experiences via online simulation (default)
     - a local directory or file glob expression (e.g., "/tmp/*.json")
     - a list of individual file paths/URIs (e.g., ["/tmp/1.json",
       "s3://bucket/2.json"])
     - a dict with string keys and sampling probabilities as values (e.g.,
       {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2}).
     - a function that returns a rllib.offline.InputReader
    """,
    "input_evaluation": """\
    Specify how to evaluate the current policy. This only has an effect when
    reading offline experiences. Available options:
     - "wis": the weighted step-wise importance sampling estimator.
     - "is": the step-wise importance sampling estimator.
     - "simulation": run the environment in the background, but use
       this data for evaluation only and not for learning.
    """,
    "postprocess_inputs": """\
    Whether to run postprocess_trajectory() on the trajectory fragments from
    offline inputs. Note that postprocessing will be done using the *current*
    policy, not the *behavior* policy, which is typically undesirable for
    on-policy algorithms.
    """,
    "shuffle_buffer_size": """\
    If positive, input batches will be shuffled via a sliding window buffer
    of this number of batches. Use this if the input data is not in random
    enough order. Input is delayed until the shuffle buffer is filled.
    """,
    "output": """\
    Specify where experiences should be saved:
     - None: don't save any experiences
     - "logdir" to save to the agent log dir
     - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
     - a function that returns a rllib.offline.OutputWriter
    """,
    "output_compress_columns": """\
    What sample batch columns to LZ4 compress in the output data.
    """,
    "output_max_file_size": """\
    Max output file size before rolling over to a new file.
    """,
    # === Settings for Multi-Agent Environments ===
    "multiagent": {
        "__help__": """\
        === Settings for Multi-Agent Environments ===
        """,
        "policies": """\
        Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        of (policy_cls, obs_space, act_space, config). This defines the
        observation and action spaces of the policies and any extra config.
        """,
        "policy_mapping_fn": """\
        Function mapping agent ids to policy ids.
        """,
        "policies_to_train": """\
        Optional list of policies to train, or None for all policies.
        """,
        "observation_fn": """\
        Optional function that can be used to enhance the local agent
        observations to include more state.
        See rllib/evaluation/observation_function.py for more info.
        """,
        "replay_mode": """\
        When replay_mode=lockstep, RLlib will replay all the agent
        transitions at a particular timestep together in a batch. This allows
        the policy to implement differentiable shared computations between
        agents it controls at that timestep. When replay_mode=independent,
        transitions are replayed independently per policy.
        """,
    },
    "use_pytorch": """\
    Deprecated; replaced by `framework=torch`.
    """,
    "eager": """\
    Deprecated; replaced by `framework=tfe`.
    """,
}

Config = Dict[str, Union[Json, "Config"]]
Info = Dict[str, Union[str, "Info"]]


def dedent_info_dict(info: Info) -> Info:
    """Remove any common leading whitespaces from every help text in info."""
    return tree.map_structure(
        (lambda x: dedent(x).rstrip() if isinstance(x, str) else x), info
    )


def with_rllib_info(info: Info) -> Info:
    """Merge info with RLlib's common parameters' info."""
    info = with_base_config(COMMON_INFO, info)
    return dedent_info_dict(info)


class MissingConfigInfoError(Exception):
    """Exception raised for undocumented config parameter.

    Args:
        key: Name of config parameter which is missing in info dict
    """

    def __init__(self, key: str):
        super().__init__(f"Info does not document {key} config parameter(s).")


class ExtraConfigInfoError(Exception):
    """Exception raised for documented inexistent config parameter.

    Args:
        key: Name of info key which does not exist in config
    """

    def __init__(self, key: str):
        super().__init__(f"Info key(s) {key} does no exist in config dict.")


def recursive_check_info(
    config: Config,
    info: Info,
    allow_new_subkey_list: Optional[List[str]] = None,
    prefix: str = "",
):
    """Recursively check if all keys in config are documented in info.

    Args:
        config: Configuration dictionary with default values
        info: Help dictionary for parameters
        allow_new_subkey_list: List of sub-dictionary keys with arbitrary items.
            Skips recursive check of these keys.
        prefix: String for keeping track of parent config keys. Used when
            raising errors.

    Raises:
        MissingConfigInfoError: If a parameter is not documented
        ExtraConfigInfoError: If info documents an inexistent parameter
    """
    allow_new_subkey_list = allow_new_subkey_list or []

    config_keys = set(config.keys())
    info_keys = set(info.keys())

    missing = config_keys.difference(info_keys)
    if missing:
        raise MissingConfigInfoError(prefix + f"{missing}")

    extra = info_keys.difference(config_keys)
    if extra:
        raise ExtraConfigInfoError(prefix + f"{extra}")

    for key in config.keys():
        config_, info_ = config[key], info[key]
        if isinstance(config_, dict) and isinstance(info_, dict):
            if "__help__" not in info_:
                raise MissingConfigInfoError(prefix + key)
            if key in allow_new_subkey_list:
                continue
            recursive_check_info(config_, info_, prefix=key + "/")
