import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        This is a modified version of the [notebook I made](file:../tutorial)
        of the [PyClaw tutorial](http://www.clawpack.org/pyclaw/tutorial.html).
        But instead of acoustic waves, I am going to try and use the euler solvers and set up a shock.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## The Solver

        Use the python version of the Roe solver for now. And the boundary conditions seem to be outflow. I am taking the ones from the Sod shock problem.
        """
    )
    return


@app.cell
def __(pyclaw, riemann):
    solver = pyclaw.ClawSolver1D(riemann.euler_1D_py.euler_roe_1D)
    solver.kernel_language = "Python"
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver
    return (solver,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## The Domain

        In the shocktube.py example, we have an extra layer of objects. Instead of just instantiating the Domain, we have a Dimension inside a Domain
        """
    )
    return


@app.cell
def __(pyclaw):
    _mx = 800
    domain = pyclaw.Domain([pyclaw.Dimension(-1.0, 1.0, _mx, name="x")])
    domain.patch
    return (domain,)


@app.cell
def __():
    # solution = pyclaw.Solution(solver.num_eqn, domain)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## The State and its initial conditions

        This is also done differently to in the acoustics example. Rather than instantiating a Solution and then taking the .state property of that, we first instantiate the State, 

        We initialize a step function in the density and pressure for the shock tube problem. I use lots of local variables with the underscore prefix to avoid polluting the global namespace of the notebook. *It might make more sense to write this as function, so the underscores would become unnecessary.*
        """
    )
    return


@app.cell
def __(i_density, i_energy, i_momentum, np):
    def shock_tube_initialize_state(
        state, rho_l=1.0, rho_r=1.0 / 8.0, p_l=1.0, p_r=0.1
    ):
        """Set up initial conditions for shock tube problem with left and right density and pressure.

        The state.q variables are set in place.
        """
        gamma1 = state.problem_data["gamma1"]
        # Cell center coordinates
        x = state.grid.x.centers
        # Initialize the two sides
        state.q[i_density, :] = np.where(x < 0.0, rho_l, rho_r)
        state.q[i_momentum, :] = 0.0
        velocity = state.q[i_momentum, :] / state.q[i_density, :]
        pressure = np.where(x < 0.0, p_l, p_r)
        # Thermal plus kinetic energy
        state.q[i_energy, :] = (
            pressure / gamma1 + 0.5 * state.q[i_density, :] * velocity**2
        )

        return None
    return (shock_tube_initialize_state,)


@app.cell
def __(GAMMA, domain, pyclaw, shock_tube_initialize_state, solver):
    state = pyclaw.State(domain, solver.num_eqn)
    # Adiabatic to start with
    # _gamma = 5.0 / 3.0
    _gamma = GAMMA.value
    state.problem_data["gamma"] = _gamma
    state.problem_data["gamma1"] = _gamma - 1.0
    state.problem_data["efix"] = False

    shock_tube_initialize_state(state, p_l=4, rho_r=0.04, p_r=0.01)
    state
    return (state,)


@app.cell
def __(mo):
    mo.md(r"""We will also write a function here to convert conserved variables to primitives, which we will use later in the plotting""")
    return


@app.cell
def __(i_density, i_energy, i_momentum, np):
    def primitives(state):
        "Return a state's primitive variables: density, velocity, pressure"
        density = state.q[i_density, :]
        velocity = state.q[i_momentum, :] / density
        pressure = state.problem_data["gamma1"] * (
            state.q[i_energy, :] - 0.5 * density * velocity**2
        )
        return density, velocity, pressure


    def sound_speed(state):
        density, velocity, pressure = primitives(state)
        return np.sqrt(state.problem_data["gamma"] * pressure / density)
    return primitives, sound_speed


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### The controller""")
    return


@app.cell
def __(domain, pyclaw, solver, state):
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.solver = solver
    controller.tfinal = 0.4
    controller.num_output_times = 40

    # Make sure we keep the intermediate time solutions
    controller.keep_copy = True

    status = controller.run()
    status
    return controller, status


@app.cell
def __(mo):
    mo.md(r""" """)
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Plot solution

        Individual timestep solutions are in `controller.frames`. I will try and set up a slider to control which one is plotted.
        """
    )
    return


@app.cell
def __(GAMMA, controller, mo):
    _ = GAMMA
    itime = mo.ui.number(
        start=0,
        stop=controller.num_output_times,
        step=1,
        label="Timestep",
        debounce=True,
    )
    return (itime,)


@app.cell
def __(mo):
    GAMMA = mo.ui.number(
        start=1.001,
        stop=2.0,
        step=0.001,
        value=1.667,
        label=r"Adiabatic index, $\gamma$",
        debounce=True,
    )
    return (GAMMA,)


@app.cell
def __(controller, itime, np, pd, primitives):
    _state = controller.frames[itime.value].state
    _d, _v, _p = primitives(_state)
    _x = _state.grid.x.centers


    def tidy(col, step=80, precision=3):
        return np.round(col[slice(None, None, 80)], precision)


    df = pd.DataFrame(
        {
            "x": tidy(_x, precision=2),
            "density": tidy(_d),
            "velocity": tidy(_v),
            "pressure": tidy(_p),
        }
    )
    df
    return df, tidy


@app.cell
def __(GAMMA, controller, itime, mo, plt, primitives, sound_speed):
    fig, ax = plt.subplots()
    # get thte state at the current time
    _state = controller.frames[itime.value].state
    _x = _state.grid.x.centers
    _d, _v, _p = primitives(_state)
    _a = sound_speed(_state)

    _ke = 0.5 * _d * _v**2
    _ie = _p / _state.problem_data["gamma1"]
    _m = _v / _a

    ax.plot(_x, _d, label="density")
    ax.plot(_x, _v, label="velocity")
    ax.plot(_x, _p, label="pressure")
    ax.plot(_x, _a, label="sound speed")
    ax.plot(_x, _m, label="mach")


    ax.axvline(0.0, color="k", ls="dotted")
    ax.legend(loc="upper right", ncol=2)
    ax.set(
        ylim=[-0.1, 5.1],
    )
    mo.hstack(
        [
            fig,
            mo.vstack(
                [
                    itime,
                    mo.md(f"Time = {_state.t:.1f}"),
                    GAMMA,
                ]
            ),
        ]
    )
    return ax, fig


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Imports

        Put them at the end because we can
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    from clawpack import pyclaw
    from clawpack import riemann

    # Indices of each conserved variable in the solution arrays
    from clawpack.riemann.euler_with_efix_1D_constants import (
        density as i_density,
        momentum as i_momentum,
        energy as i_energy,
    )
    from clawpack.pyclaw import plot
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    return (
        i_density,
        i_energy,
        i_momentum,
        np,
        pd,
        plot,
        plt,
        pyclaw,
        riemann,
        sns,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
