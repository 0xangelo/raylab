# pylint:disable=missing-docstring
import timeit
from textwrap import dedent


def main():
    # code = """
    # with torch.jit.optimized_execution(True):
    #     params = module.model.params(obs, act)
    # """
    # code = """
    # params = module.model.params(obs, act)
    # """
    # code = """
    # torch.cat([m(obs, act) for m in module.critics], dim=-1)
    # """
    # code = """
    # module.actor.rsample(obs)
    # """
    # code = """
    # module.actor.reproduce(obs, act)
    # """
    code = """
    module.model.rsample(obs, act)
    """
    code = dedent(code)

    base_setup = """\
    import logging
    import torch
    import gym
    import gym.spaces as spaces
    from raylab.modules.catalog import SVGModule as mod_cls

    gym.logger.set_level(logging.ERROR)
    obs_space = spaces.Box(-float("inf"), float("inf"), shape=(8,))
    action_space = spaces.Box(-1., 1., shape=(3,))
    obs = torch.randn(10, 8)
    act = torch.randn(10, 3)
    """

    eager_setup = dedent(
        base_setup
        + """
    module = mod_cls(obs_space, action_space, {"torch_script": False})
    """
    )

    script_setup = dedent(
        base_setup
        + """
    module = torch.jit.script(
        mod_cls(obs_space, action_space, {"torch_script": True})
    )
    """
    )

    times = list(
        zip(
            timeit.repeat(code, setup=eager_setup, number=1000),
            timeit.repeat(code, setup=script_setup, number=1000),
        )
    )
    print("Average eager time:", sum(e for e, s in times) / len(times))
    print("Average script time:", sum(s for e, s in times) / len(times))
    print("Average time difference:", sum(e - s for e, s in times) / len(times))


if __name__ == "__main__":
    main()
