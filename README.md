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
    ├── envs            # Gym environments
    ├── logger          # Tune loggers
    ├── policy          # Extensions and customizations of ray.rllib.policy submodules
    ├── utils           # miscellaneous utilities
    └── viskit          # visualization tools based on rllab
    
## Running experiments
One can run Tune experiments locally using `scripts/tune_experiment.py`.
```bash
python scripts/tune_experiment.py --help
```
An example is included in `examples/naf_exploration_experiment.py`.

One can also use `scripts/train.py`, which wraps 
[`rllib train`](https://ray.readthedocs.io/en/latest/rllib-training.html#rllib-training-apis)
so as to register custom algorithms and environments beforehand.

## Evaluating agents
It is recommended to use `scripts/rollout.py`, which wraps 
[`rllib rollout`](https://ray.readthedocs.io/en/latest/rllib-training.html#evaluating-trained-policies)
so as to register custom algorithms and environments beforehand.

## Visualizing results
Tune logs metrics in a Tensorboard compatible format. To generate `matplotlib` plots, check out the `viskit` entrypoint 
installed alongside `raylab`.
```bash
viskit --help
```
