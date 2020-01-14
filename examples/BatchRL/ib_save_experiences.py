"""Collect and save experiencies in IndustrialBenchmark with a behaviour policy."""
from tqdm import trange
import click
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from raylab.envs.registry import _industrial_benchmark_maker

from ib_behavior_policy import IBBehaviorPolicy


@click.command()
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    default="data/",
    show_default=True,
    help="",
)
def main(local_dir):
    """Main loop based on `rllib.examples.saving_experiences`."""
    # pylint: disable=too-many-locals
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(local_dir)
    env = _industrial_benchmark_maker({"max_episode_steps": 1000})
    policy = IBBehaviorPolicy(env.observation_space, env.action_space, {})

    for eps_id in trange(100):
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        time = 0
        while not done:
            action, _, _ = policy.compute_single_action(obs, [])
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=time,
                eps_id=eps_id,
                agent_index=0,
                obs=obs,
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=new_obs,
            )
            obs = new_obs
            prev_action = action
            prev_reward = rew
            time += 1
        writer.write(batch_builder.build_and_reset())


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
