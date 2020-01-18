"""Implementation of operational costs dynamics described in section 3.B."""
import torch


COST_SETPOINT = 2.0
COST_VELOCITY = 4.0
COST_GAIN = 2.5


def current_operational_cost(state):
    """Calculate the current operational cost through equation (6) of the paper."""
    setpoint, velocity, gain = state[..., :3].chunk(3, dim=-1)

    # Calculation of equation (6)
    costs = COST_SETPOINT * setpoint + COST_VELOCITY * velocity + COST_GAIN * gain
    return torch.exp(costs / 100.0)


def update_operational_cost_history(state, old_cost):
    """
    Shift the history of operational costs and add the previously current cost.
    """
    # Update history of operational costs for future use in equation (7)
    return torch.cat(
        [state[..., :6], old_cost, state[..., 6:14], state[..., 15:]], dim=-1
    )


def convoluted_operational_cost(state):
    """Calculate the convoluted cost according to equation (7) of the paper."""
    conv = torch.as_tensor(
        [0.0, 0.0, 0.0, 0.0, 0.11111, 0.22222, 0.33333, 0.22222, 0.11111]
    )
    return state[..., 6:15].matmul(conv).unsqueeze(-1)
