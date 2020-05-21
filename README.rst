======
raylab
======


.. image:: https://img.shields.io/pypi/v/raylab.svg
        :target: https://pypi.python.org/pypi/raylab

.. image:: https://img.shields.io/travis/angelolovatto/raylab.svg
        :target: https://travis-ci.com/angelolovatto/raylab

.. image:: https://readthedocs.org/projects/raylab/badge/?version=latest
        :target: https://raylab.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/angelolovatto/raylab/shield.svg
     :target: https://pyup.io/repos/github/angelolovatto/raylab/
     :alt: Updates



Reinforcement learning algorithms in `RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_ and `PyTorch <https://pytorch.org>`_.


* Free software: MIT license
* Documentation: https://raylab.readthedocs.io.


Introduction
------------

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


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
