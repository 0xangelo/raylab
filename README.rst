======
raylab
======

|PyPI| |Travis| |PyUp|

.. |PyPI| image:: https://img.shields.io/pypi/v/raylab?logo=PyPi&logoColor=white
        :alt: PyPI

.. |Travis| image:: https://img.shields.io/travis/com/angelolovatto/raylab?logo=travis-ci&logoColor=important
        :alt: Travis (.com)

.. |PyUp| image:: https://pyup.io/repos/github/angelolovatto/raylab/shield.svg
     :target: https://pyup.io/repos/github/angelolovatto/raylab/
     :alt: Updates


Reinforcement learning algorithms in `RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_ and `PyTorch <https://pytorch.org>`_.


Introduction
------------

Raylab provides agents and environments to be used with a normal RLlib/Tune setup.

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
                     local_dir=...,
                     stop={"timesteps_total": 100000},
                     config={
                         "env": "CartPoleSwingUp-v0",
                         "exploration_config": {
                             "type": tune.grid_search([
                                 "raylab.utils.exploration.GaussianNoise",
                                 "raylab.utils.exploration.ParameterNoise"
                             ])
                         }
                         ...
                     },
                 )

             if __name__ == "__main__":
                 main()


One can then visualize the results using `raylab dashboard`

.. image:: https://i.imgur.com/bVc6WC5.png
        :align: center


Installation
------------

.. code:: bash

          pip install raylab


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


.. _`Actor Critic using Kronecker-factored Trust Region`: https://arxiv.org/abs/1708.05144
.. _`Trust Region Policy Optimization`: http://proceedings.mlr.press/v37/schulman15.html
.. _`Normalized Advantage Function`: http://proceedings.mlr.press/v48/gu16.html
.. _`Stochastic Value Gradients`: http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients
.. _`Soft Actor-Critic`: http://proceedings.mlr.press/v80/haarnoja18b.html
.. _`Streamlined Off-Policy`: https://arxiv.org/abs/1910.02208


Command-line interface
----------------------

For a high-level description of the available utilities, run `raylab --help`

.. code:: bash

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


Packages
--------

The project is structured as follows
::

    raylab
    ├── agents            # Trainer and Policy classes
    ├── cli               # Command line utilities
    ├── distributions     # Extendend and additional PyTorch distributions
    ├── envs              # Gym environments
    ├── logger            # Tune loggers
    ├── modules           # PyTorch neural network modules for algorithms
        ├── basic         # Building blocks for neural networks
        ├── flows         # Normalizing Flow modules
        ├── distributions # TorchScript compatible distribution modules
    ├── policy            # Extensions and customizations of RLlib's policy API
    ├── utils             # miscellaneous utilities



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
