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
```bash
raylab --help
```

### Running experiments
One can run Tune experiments locally using `raylab experiment`.
```bash
raylab experiment --help
```
An example is included in `examples/naf_exploration_experiment.py`.

One can also use `scripts/train.py`, which wraps
[`rllib train`](https://ray.readthedocs.io/en/latest/rllib-training.html#rllib-training-apis)
so as to register custom algorithms and environments beforehand.

### Evaluating agents
To load a checkopoint and simulate an agent, use
```bash
raylab rollout
```

If one prefers rllib's interface, it is recommended to use `scripts/rollout.py`, which wraps
[`rllib rollout`](https://ray.readthedocs.io/en/latest/rllib-training.html#evaluating-trained-policies)
so as to register custom algorithms and environments beforehand.

### Visualizing results
Tune logs metrics in a Tensorboard compatible format. To generate `matplotlib` plots, check out the `plot` and `plot-export` subcommands under `raylab`.
```bash
raylab plot --help
raylab plot-export --help
```
These use [seaborn](https://seaborn.pydata.org/generated/seaborn.lineplot.html)'s `lineplot` to generate time series comparisons.

## Algorithms

* [NAF](http://proceedings.mlr.press/v48/gu16.html)
* [SVG(inf) and SVG(1)](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients)
* [SAC](http://proceedings.mlr.press/v80/haarnoja18b.html)
* [SOP](https://arxiv.org/abs/1910.02208)
