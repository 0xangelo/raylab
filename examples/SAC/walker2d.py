from cheetah import get_config as cheetah_config


def get_config() -> dict:
    from ray import tune

    base = cheetah_config()
    base["env"] = "Walker2d-v3"
    # base["policy"]["std_obs"] = tune.grid_search([True, False])
    return base
