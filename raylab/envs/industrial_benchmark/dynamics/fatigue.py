"""Implementation of fatigue dynamics described in section 3.C."""
import torch
import torch.distributions as dists


ACTION_TOLERANCE = 0.05
FATIGUE_AMPLIFICATION = 1.1
FATIGUE_AMPLIFICATION_MAX = 5.0
FATIGUE_AMPLIFICATION_START = 1.2


def effective_velocity(velocity, gain, setpoint):
    """Equation 24."""
    maximum = unscaled_eff_velocity(100.0, 0.0, setpoint)
    minimum = unscaled_eff_velocity(0.0, 100.0, setpoint)
    return (unscaled_eff_velocity(velocity, gain, setpoint) - minimum) / (
        maximum - minimum
    )


def effective_gain(gain, setpoint):
    """Equation 25."""
    maximum = unscaled_eff_gain(100, setpoint)
    minimum = unscaled_eff_gain(0, setpoint)
    return (unscaled_eff_gain(gain, setpoint) - minimum) / (maximum - minimum)


def unscaled_eff_velocity(velocity, gain, setpoint):
    """Equation 26."""
    return (gain + setpoint + 2.0) / (velocity - setpoint + 101.0)


def unscaled_eff_gain(gain, setpoint):
    """Equation 27."""
    return 1.0 / (gain + setpoint + 1)


def sample_noise_variables(eff_velocity, eff_gain):
    """Equations 28, 29."""
    # Noise variables described after equation (27)
    eta_e_dist = dists.Exponential(torch.empty_like(eff_gain).fill_(0.1))
    eta_ge = eta_e_dist.rsample()
    eta_ve = eta_e_dist.rsample()
    # Apply the logistic fuction to exponential variables
    eta_ge = 2.0 * (1.0 / (1.0 + torch.exp(-eta_ge)) - 0.5)
    eta_ve = 2.0 * (1.0 / (1.0 + torch.exp(-eta_ve)) - 0.5)

    eta_u_dist = dists.Uniform(torch.zeros_like(eff_gain), torch.ones_like(eff_gain))
    eta_gu = eta_u_dist.rsample()
    eta_vu = eta_u_dist.rsample()

    eta_gb = dists.Bernoulli(torch.clamp(eff_gain, 0.001, 0.999)).sample()
    eta_vb = dists.Bernoulli(torch.clamp(eff_velocity, 0.001, 0.999)).sample()

    # Equations (28, 29)
    noise_velocity = eta_ve + (1 - eta_ve) * eta_vu * eta_vb * eff_velocity
    noise_gain = eta_ge + (1 - eta_ge) * eta_gu * eta_gb * eff_gain
    return noise_velocity, noise_gain


def update_hidden_velocity(hidden_velocity, eff_velocity, noise_velocity):
    """Equation 30."""
    return torch.where(
        eff_velocity <= ACTION_TOLERANCE,
        eff_velocity,
        torch.where(
            hidden_velocity >= FATIGUE_AMPLIFICATION_START,
            torch.min(
                torch.as_tensor(FATIGUE_AMPLIFICATION_MAX),
                FATIGUE_AMPLIFICATION * hidden_velocity,
            ),
            0.9 * hidden_velocity + noise_velocity / 3.0,
        ),
    )


def update_hidden_gain(hidden_gain, eff_gain, noise_gain):
    """Equation 31."""
    return torch.where(
        eff_gain <= ACTION_TOLERANCE,
        eff_gain,
        torch.where(
            hidden_gain >= FATIGUE_AMPLIFICATION_START,
            torch.min(
                torch.as_tensor(FATIGUE_AMPLIFICATION_MAX),
                FATIGUE_AMPLIFICATION * hidden_gain,
            ),
            0.9 * hidden_gain + noise_gain / 3.0,
        ),
    )


def sample_alpha(hidden_velocity, noise_velocity, hidden_gain, noise_gain):
    """Equation 23."""
    return torch.where(
        torch.max(hidden_velocity, hidden_gain) >= FATIGUE_AMPLIFICATION_MAX,
        1.0 / (1.0 + torch.exp(-(torch.randn_like(hidden_velocity) * 0.4 + 2.4))),
        torch.max(noise_velocity, noise_gain),
    )


def basic_fatigue(velocity, gain):
    """Equation 21."""
    return torch.max(
        torch.as_tensor(0.0), ((30000.0 / ((5 * velocity) + 100)) - 0.01 * (gain ** 2))
    )


def fatigue(basic_fatigue_, alpha):
    """Equation 22."""
    return (basic_fatigue_ * (1 + 2 * alpha)) / 3.0
