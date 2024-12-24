"""
Various routines for working with radiative shock simulations with clawpack
"""
import numpy as np
from clawpack import pyclaw
from clawpack.riemann.euler_with_efix_1D_constants import (
    density as i_density,
    momentum as i_momentum,
    energy as i_energy,
)

# Default adiabatic index
GAMMA = 5.0 / 3.0


def get_variables_dict(state: pyclaw.State):
    "Return a dict of all primitive and derived variables"
    gamma = state.problem_data.get("gamma", GAMMA)
    gamma1 = state.problem_data.get("gamma1", GAMMA - 1.0)
    density = state.q[i_density, :]
    velocity = state.q[i_momentum, :] / density
    pressure = gamma1 * (
        state.q[i_energy, :] - 0.5 * density * velocity**2
    )
    temperature = pressure / density
    sound_speed = np.sqrt(gamma * pressure / density)
    mach_number = np.abs(velocity / sound_speed)
    return {
        "density": density,
        "velocity": velocity,
        "pressure": pressure,
        "temperature": temperature,
        "sound speed": sound_speed,
        "mach number": mach_number,
    }


class ShockConditions:
    """Jump conditions for a radiative shock

    Terminology:
        0 = upstream state
        1 = immediate post-shock state
        2 = post-cooling state
        i = isothermal
        a = adiabatic
    """

    # Class parameters
    GAMMA = 5.0 / 3.0

    def __init__(self, mach_i_0):
        "Radiative shock with isothermal Mach number `mach_i_0`"
        assert mach_i_0 >= 1.0, "Mach number cannot be less than unity"
        self.mach_i_0 = mach_i_0
        G = self.GAMMA
        # Convenience constants: G13 = 1/3 and G5 = 5 for GAMMA = 5/3
        G13 = (G - 1) / 2
        G5 = 2 * G / (G - 1)
        # Adiabatic Mach number
        self.mach_a_0 = mach_i_0 / np.sqrt(G)
        M0sq = self.mach_a_0**2  # Save square for convenience
        # Final compression ratio: rho_2 / rho_0
        self.R_2 = mach_i_0**2
        if self.mach_a_0 >= 1.0:
            # Immediate post-shock compression ratio: rho_1 / rho_0
            self.R_1 = (G + 1) * M0sq / ((G - 1) * M0sq + 2)
            # Immediate post-shock temperature
            self.T_1 = (1 + G13 * M0sq) * (G5 - 1 / M0sq) / (G13 + G5)
            # Immediate post-shock pressure ratio: p_1 / p_0
            self.p_1 = (2 * G * M0sq - (G - 1)) / (G + 1)
        else:
            # Case of quasi-shocks
            self.R_1 = self.T_1 = self.p_1 = 1.0
        # Velocity jump across shock in units of isothermal sound speed
        self.U_jump = self.mach_i_0 - 1 / self.mach_i_0
        # Check that pressure jump is density jump times temperature jump
        assert np.isclose(self.p_1, self.R_1 * self.T_1)
