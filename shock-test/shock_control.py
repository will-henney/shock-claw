import sys
import typer
from clawpack import pyclaw, riemann
from clawpack.riemann.euler_with_efix_1D_constants import (
    density as i_density,
    momentum as i_momentum,
    energy as i_energy,
)
import numpy as np
import shock_utils
import cooling_function


def shock_tube_initialize_state(
    state,
    rho_l=1.0,
    rho_r=1.0,
    v_l=1.0,
    v_r=-1.0,
    p_l=1.0,
    p_r=1.0,
):
    """Initial conditions for shock tube problem with left/right density, velocity, pressure.

    The state.q variables are set in place.
    """
    gamma1 = state.problem_data["gamma1"]
    # Cell center coordinates
    x = state.grid.x.centers
    # Initialize the two sides
    state.q[i_density, :] = np.where(x < 0.0, rho_l, rho_r)
    state.q[i_momentum, :] = np.where(x < 0.0, rho_l * v_l, rho_r * v_r)
    velocity = state.q[i_momentum, :] / state.q[i_density, :]
    pressure = np.where(x < 0.0, p_l, p_r)
    # Thermal plus kinetic energy
    state.q[i_energy, :] = pressure / gamma1 + 0.5 * state.q[i_density, :] * velocity**2

    return None


def main(
    mach_number: float = typer.Option(
        2.0, help="The isothermal Mach number of the shock"
    ),
    ncells: int = typer.Option(800, help="Number of cells in the domain"),
    cool_rate: float = typer.Option(10.0, help="Post-shock cooling rate in code units"),
    cool_slope: float = typer.Option(
        2.3, help="Power-law slope of the cooling function"
    ),
    output_format: str = typer.Option("hdf5", help="Output format for the simulation"),
):
    """Carry out colliding stream simulation with radiative cooling."""
    # Get the shock jump conditions
    shock = shock_utils.ShockConditions(mach_number)

    model_id = (
        f"shock-Ma-{mach_number:.1f}"
        f"-Lambda-{cool_rate:04.1f}"
        f"-q-{cool_slope:.1f}"
        f"-N-{ncells:04d}"
    )

    print(model_id)

    # Solver and boundary conditions
    solver = pyclaw.ClawSolver1D(riemann.euler_1D_py.euler_roe_1D)
    solver.kernel_language = "Python"
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    # Domain for colliding stream simulation
    domain = pyclaw.Domain([pyclaw.Dimension(-1.0, 1.0, ncells, name="x")])
    # Initialize state
    state = pyclaw.State(domain, solver.num_eqn)
    # Adiabatic to start with
    state.problem_data["gamma"] = 5 / 3
    state.problem_data["gamma1"] = 2 / 3
    state.problem_data["efix"] = False

    # Parameters for the cooling
    state.problem_data["cool_method"] = "second"
    state.problem_data["cool_rate"] = cool_rate
    state.problem_data["cool_slope"] = cool_slope
    state.problem_data["T_eq"] = 1.0

    # Use oppositely directed velocities and aim for a post-cooled density of 1
    shock_tube_initialize_state(
        state,
        rho_l=1 / shock.R_2,
        p_l=1 / shock.R_2,
        v_l=shock.U_jump,
        rho_r=1 / shock.R_2,
        p_r=1 / shock.R_2,
        v_r=-shock.U_jump,
    )

    # Set up the controller
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.solver = solver
    controller.tfinal = 1.0
    controller.num_output_times = 100

    # Make sure we keep the intermediate time solutions
    controller.keep_copy = True
    controller.output_format = output_format
    # Bad idea to change from default of "fort" due to bug
    # controller.output_file_prefix = "shock"
    controller.outdir = f"shock-output/{model_id}"
    status = controller.run()


if __name__ == "__main__":
    typer.run(main)
