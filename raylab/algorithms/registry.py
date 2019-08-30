"""Registry of algorithm names for `rllib train --run=<alg_name>`"""


def _import_ppo():
    from raylab.algorithms.ppo.ppo import PPOTrainer

    return PPOTrainer


def _import_td3():
    from raylab.algorithms.ddpg.td3 import TD3Trainer

    return TD3Trainer


LOCAL_ALGORITHMS = {"contrib/PPO": _import_ppo, "contrib/TD3": _import_td3}
