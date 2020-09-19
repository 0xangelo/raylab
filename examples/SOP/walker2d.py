from cheetah import get_config as cheetah_config


def get_config() -> dict:
    base = cheetah_config()
    base["env"] = "Walker2d-v3"
    return base
