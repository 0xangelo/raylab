======
raylab
======

|PyPI| |Tests| |Dependabot| |License| |CodeStyle|

.. |PyPI| image:: https://img.shields.io/pypi/v/raylab?logo=PyPi&logoColor=white&color=blue
      :alt: PyPI

.. |Tests| image:: https://img.shields.io/github/workflow/status/angelolovatto/raylab/Poetry%20package?label=tests&logo=GitHub
       :alt: GitHub Workflow Status

.. |Dependabot| image:: https://api.dependabot.com/badges/status?host=github&repo=angelolovatto/raylab
        :target: https://dependabot.com

.. |License| image:: https://img.shields.io/github/license/angelolovatto/raylab?color=blueviolet&logo=github
         :alt: GitHub

.. |CodeStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
           :target: https://github.com/psf/black


Reinforcement learning algorithms in `RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_
and `PyTorch <https://pytorch.org>`_.


Installation
------------

.. code:: bash

          pip install raylab


Quickstart
----------

Raylab provides agents and environments to be used with a normal RLlib/Tune setup.
You can an agent's name (from the `Algorithms`_ section) to :code:`raylab info list` to list its top-level configurations:

.. code-block:: zsh

    raylab info list SoftAC

.. code-block::

    learning_starts: 0
        Hold this number of timesteps before first training operation.
    policy: {}
        Sub-configurations for the policy class.
    wandb: {}
        Configs for integration with Weights & Biases.

        Accepts arbitrary keyword arguments to pass to `wandb.init`.
        The defaults for `wandb.init` are:
        * name: `_name` property of the trainer.
        * config: full `config` attribute of the trainer
        * config_exclude_keys: `wandb` and `callbacks` configs
        * reinit: True

        Don't forget to:
          * install `wandb` via pip
          * login to W&B with the appropriate API key for your
            team/project.
          * set the `wandb/project` name in the config dict

        Check out the Quickstart for more information:
        `https://docs.wandb.com/quickstart`

You can add the :code:`--rllib` flag to get the descriptions for all the options common to RLlib agents
(or :code:`Trainer`\s)

Launching experiments can be done via the command line using :code:`raylab experiment` passing a file path
with an agent's configuration through the :code:`--config` flag.
The following command uses the cartpole `example <examples/PG/cartpole_defaults.py>`_ configuration file
to launch an experiment using the vanilla Policy Gradient agent from the RLlib library.

.. code-block:: zsh

    raylab experiment PG --name PG -s training_iteration 10 --config examples/PG/cartpole_defaults.py

You can also launch an experiment from a Python script normally using Ray and Tune.
The following shows how you may use Raylab to perform an experiment comparing different
types of exploration for the NAF agent.

.. code-block:: python

             import ray
             from ray import tune
             import raylab

             def main():
                 raylab.register_all_agents()
                 raylab.register_all_environments()
                 ray.init()
                 tune.run(
                     "NAF",
                     local_dir="data/NAF",
                     stop={"timesteps_total": 100000},
                     config={
                         "env": "CartPoleSwingUp-v0",
                         "exploration_config": {
                             "type": tune.grid_search([
                                 "raylab.utils.exploration.GaussianNoise",
                                 "raylab.utils.exploration.ParameterNoise"
                             ])
                         }
                     },
                     num_samples=10,
                 )

             if __name__ == "__main__":
                 main()


One can then visualize the results using :code:`raylab dashboard`, passing the :code:`local_dir` used in the
experiment. The dashboard lets you filter and group results in a quick way.

.. code-block:: zsh

    raylab dashboard data/NAF/


.. image:: https://i.imgur.com/bVc6WC5.png
        :align: center


You can find the best checkpoint according to a metric (:code:`episode_reward_mean` by default)
using :code:`raylab find-best`.

.. code-block:: zsh

    raylab find-best data/NAF/

Finally, you can pass a checkpoint to :code:`raylab rollout` to see the returns collected by the agent and
render it if the environment supports a visual :code:`render()` method. For example, you
can use the output of the :code:`find-best` command to see the best agent in action.


.. code-block:: zsh

    raylab rollout $(raylab find-best data/NAF/) --agent NAF


Algorithms
----------

+--------------------------------------------------------+-------------------------+
| Paper                                                  | Agent Name              |
+--------------------------------------------------------+-------------------------+
| `Actor Critic using Kronecker-factored Trust Region`_  | ACKTR                   |
+--------------------------------------------------------+-------------------------+
| `Trust Region Policy Optimization`_                    | TRPO                    |
+--------------------------------------------------------+-------------------------+
| `Normalized Advantage Function`_                       | NAF                     |
+--------------------------------------------------------+-------------------------+
| `Stochastic Value Gradients`_                          | SVG(inf)/SVG(1)/SoftSVG |
+--------------------------------------------------------+-------------------------+
| `Soft Actor-Critic`_                                   | SoftAC                  |
+--------------------------------------------------------+-------------------------+
| `Streamlined Off-Policy`_ (DDPG)                       | SOP                     |
+--------------------------------------------------------+-------------------------+
| `Model-Based Policy Optimization`_                     | MBPO                    |
+--------------------------------------------------------+-------------------------+
| `Model-based Action-Gradient-Estimator`_               | MAGE                    |
+--------------------------------------------------------+-------------------------+


.. _`Actor Critic using Kronecker-factored Trust Region`: https://arxiv.org/abs/1708.05144
.. _`Trust Region Policy Optimization`: http://proceedings.mlr.press/v37/schulman15.html
.. _`Normalized Advantage Function`: http://proceedings.mlr.press/v48/gu16.html
.. _`Stochastic Value Gradients`: http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients
.. _`Soft Actor-Critic`: http://proceedings.mlr.press/v80/haarnoja18b.html
.. _`Model-Based Policy Optimization`: http://arxiv.org/abs/1906.08253
.. _`Streamlined Off-Policy`: https://arxiv.org/abs/1910.02208
.. _`Model-based Action-Gradient-Estimator`: https://arxiv.org/abs/2004.14309


Command-line interface
----------------------

.. role:: bash(code)
   :language: bash

For a high-level description of the available utilities, run :bash:`raylab --help`

.. code:: bash

    Usage: raylab [OPTIONS] COMMAND [ARGS]...

      RayLab: Reinforcement learning algorithms in RLlib.

    Options:
      --help  Show this message and exit.

    Commands:
      dashboard    Launch the experiment dashboard to monitor training progress.
      episodes     Launch the episode dashboard to monitor state and action...
      experiment   Launch a Tune experiment from a config file.
      find-best    Find the best experiment checkpoint as measured by a metric.
      info         View information about an agent's config parameters.
      rollout      Wrap `rllib rollout` with customized options.
      test-module  Launch dashboard to test generative models from a checkpoint.


Packages
--------

The project is structured as follows
::

    raylab
    |-- agents            # Trainer and Policy classes
    |-- cli               # Command line utilities
    |-- envs              # Gym environment registry and utilities
    |-- logger            # Tune loggers
    |-- policy            # Extensions and customizations of RLlib's policy API
    |   |-- losses        # RL loss functions
    |   |-- modules       # PyTorch neural network modules for TorchPolicy
    |-- pytorch           # PyTorch extensions
    |-- utils             # miscellaneous utilities
