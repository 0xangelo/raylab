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
    return state[..., 6:15].matmul(conv)


def update_goldstone(state, gs_env, effective_shift):
    """Calculate the domain, response, direction, and MisCalibration.

    Computes equations (9), (10), (11), and (12) of the paper under the
    'Dynamics of mis-calibration' section.
    """
    domain, system_response, phi_idx = state[..., -5:-2].chunk(3, dim=-1)

    reward, domain, phi_idx, system_response = gs_env.state_transition(
        domain.sign(), phi_idx, system_response.sign(), effective_shift
    )
    next_state = torch.cat(
        [state[..., :-5], domain, system_response, phi_idx, state[..., -2:]], dim=-1
    )
    miscalibration = -reward
    return next_state, miscalibration


def update_operational_costs(state, conv_cost, miscalibration, crgs):
    """Calculate the consumption according to equations (19) and (20)."""
    # This seems to correspond to equation (19),
    # although the minus sign is mysterious.
    hidden_cost = conv_cost - (crgs * (miscalibration - 1.0))
    # This corresponds to equation (20), although the constant 0.005 is
    # different from the 0.02 written in the paper. This might result in
    # very low observational noise
    cost = hidden_cost - torch.randn_like(hidden_cost) * (1 + 0.005 * hidden_cost)
    return torch.cat([state[..., :5], cost, state[..., 6:]], dim=-1)
