# raylab
Reinforcement learning algorithms in [RLlib](https://github.com/ray-project/ray/tree/master/rllib)

## Installation
Simply clone the repository and run
```bash
pip install -e .
```

## Structure
The project is structured as follows

    raylab
    ├── algorithms      # Trainer and Policy classes
    ├── cli             # Command line utilities
    ├── distributions   # Extendend and additional PyTorch distributions
    ├── envs            # Gym environments
    ├── logger          # Tune loggers
    ├── modules         # Additional PyTorch neural network modules
    ├── policy          # Extensions and customizations of ray.rllib.policy submodules
    ├── utils           # miscellaneous utilities

## Examples

Raylab provides algorithms and environments to be used with a normal RLlib/Tune setup. 
```python=
import ray
from ray import tune
import raylab

def main():
    raylab.register_all_agents()
    raylab.register_all_environments()
    ray.init()
    tune.run(
        "SOP",
        local_dir=...,
        stop={"timesteps_total": 100000},
        config={
            "env": "CartPoleSwingUp",
            ...
        },
    )

if __name__ == "__main__":
    main()
```

Since the setup above is likely to be repeated several times, `raylab` provides a command-line interface for [running experiments](#Running-experiments).

## Command-line interface

For a high-level description of the available utilities, run
```
Usage: raylab [OPTIONS] COMMAND [ARGS]...

  RayLab: Reinforcement learning algorithms in RLlib.

Options:
  --help  Show this message and exit.

Commands:
  dashboard    Launch the experiment dashboard to monitor training progress.
  experiment   Launch a Tune experiment from a config file.
  find-best    Find the best experiment checkpoint as measured by a metric.
  plot         Draw lineplots of the relevant variables and display them on...
  plot-export  Draw lineplots of the relevant variables and save them as...
  rollout      Simulate an agent from a given checkpoint in the desired...
```

### Running experiments
One can run Tune experiments locally using `raylab experiment`.
```
Usage: raylab experiment [OPTIONS] RUN

  Launch a Tune experiment from a config file.

Options:
  --name TEXT                     Name of experiment
  -l, --local-dir DIRECTORY       [default: data/]
  -n, --num-samples INTEGER       Number of times to sample from the
                                  hyperparameter space. Defaults to 1. If
                                  `grid_search` is provided as an argument,
                                  the grid will be repeated `num_samples` of
                                  times.  [default: 1]
  -s, --stop <TEXT INTEGER>...    The stopping criteria. The keys may be any
                                  field in the return result of 'train()',
                                  whichever is reached first. Defaults to
                                  empty dict.
  -c, --config FILE               Algorithm-specific configuration for Tune
                                  variant generation (e.g. env, hyperparams).
                                  Defaults to empty dict. Custom search
                                  algorithms may ignore this. Expects a path
                                  to a python script containing a `get_config`
                                  function.
  --checkpoint-freq INTEGER       How many training iterations between
                                  checkpoints. A value of 0 disables
                                  checkpointing.  [default: 0]
  --checkpoint-at-end BOOLEAN     Whether to checkpoint at the end of the
                                  experiment regardless of the
                                  checkpoint_freq.  [default: True]
  --object-store-memory INTEGER   The amount of memory (in bytes) to start the
                                  object store with. By default, this is
                                  capped at 20GB but can be set higher.
                                  [default: 2000000000]
  --custom-loggers / --no-custom-loggers
                                  Use custom loggers from raylab.logger.
  --tune-log-level TEXT           Logging level for the trial executor
                                  process. This is independent from each
                                  trainer's logging level.  [default: WARN]
  --help                          Show this message and exit.
```
An example is included in `examples/naf_exploration_experiment.py`.

One can also use `scripts/train.py`, which wraps 
[`rllib train`](https://ray.readthedocs.io/en/latest/rllib-training.html#rllib-training-apis)
so as to register custom algorithms and environments beforehand.

### Evaluating agents
To load a checkopoint and simulate an agent, use `raylab rollout`
```
Usage: raylab rollout [OPTIONS] CHECKPOINT

  Simulate an agent from a given checkpoint in the desired environment.

Options:
  --algo TEXT  Name of the trainable class to run.  [required]
  --env TEXT   Name of the environment to interact with, optional.
  --help       Show this message and exit.
```

If one prefers rllib's interface, it is recommended to use `scripts/rollout.py`, which wraps 
[`rllib rollout`](https://ray.readthedocs.io/en/latest/rllib-training.html#evaluating-trained-policies)
so as to register custom algorithms and environments beforehand.

### Visualizing results
Tune logs metrics in a Tensorboard compatible format. An interative dashboard for visualizing experiment results is available through `raylab dashboard` (powered by [Streamlit](http://streamlit.io) and [Bokeh](https://docs.bokeh.org/en/latest/)):
```
Usage: raylab dashboard [OPTIONS] [PATHS]...

  Launch the experiment dashboard to monitor training progress.

Options:
  --help  Show this message and exit.

```
![](https://i.imgur.com/DlOemPW.png)

To generate `matplotlib` plots, check out the `raylab plot` and `raylab plot-export`.

## Algorithms

* [NAF](http://proceedings.mlr.press/v48/gu16.html) (Normalized Advantage Function)
* [SVG(inf) and SVG(1)](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients) (Stochastic Value Gradients)
* [SAC](http://proceedings.mlr.press/v80/haarnoja18b.html) (Soft Actor-Critic)
* [SOP](https://arxiv.org/abs/1910.02208) (Streamlined Off-Policy)
