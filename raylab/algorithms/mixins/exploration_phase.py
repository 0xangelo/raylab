"""Support for updating policy exploration phase."""


class ExplorationPhaseMixin:
    """Adss method to update policies' exploration state."""

    # pylint: disable=too-few-public-methods

    def update_exploration_phase(self):
        """Signal to policies if training is still in the pure exploration phase."""
        global_timestep = self.optimizer.num_steps_sampled
        pure_expl_steps = self.config["pure_exploration_steps"]
        if pure_expl_steps:
            only_explore = global_timestep < pure_expl_steps
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, _: p.set_pure_exploration_phase(only_explore)
            )
