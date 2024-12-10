import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Colliding streams with radiative cooling, simulated with CLAWpack

        This is a modified version of the [colliding stream notebook](file:stream-collide-marimo.py), based on the [shock tube notebook](file:shock-tube-marimo.py), 
        which is itself based on an [earlier notebook I made](file:../tutorial)
        of the [PyClaw tutorial](http://www.clawpack.org/pyclaw/tutorial.html).

        This time, I will be adding my custom cooling function as a source term in the converging stream simulation. With luck, we will produce a pair of radiaitive shocks.
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
def __(cooling_function, domain, pyclaw, riemann):
    _ = domain  # re-create if domain changes
    solver = pyclaw.ClawSolver1D(riemann.euler_1D_py.euler_roe_1D)
    solver.kernel_language = "Python"
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.step_source = cooling_function.cooling_source_term_step
    solver
    return (solver,)


@app.cell(hide_code=True)
def __(mo):
    mo.callout(
        mo.md(
            "**The new thing here is that I have added my cooling function as a custom source term**"
        ),
        kind="info",
    )
    return


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
def __(NCELLS, pyclaw):
    domain = pyclaw.Domain([pyclaw.Dimension(-1.0, 1.0, NCELLS.value, name="x")])
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

        I am generalizing the shock tube initialization function to allow specifying velocities. By default, the density and pressure are the same on the two sides.
        """
    )
    return


@app.cell
def __(i_density, i_energy, i_momentum, np):
    def shock_tube_initialize_state(
        state,
        rho_l=1.0,
        rho_r=1.0,
        v_l=1.0,
        v_r=-1.0,
        p_l=1.0,
        p_r=1.0,
    ):
        """Set up initial conditions for shock tube problem with left and right density, velocity, pressure.

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
        state.q[i_energy, :] = (
            pressure / gamma1 + 0.5 * state.q[i_density, :] * velocity**2
        )

        return None
    return (shock_tube_initialize_state,)


@app.cell
def __(
    COOL_METHOD,
    COOL_RATE,
    COOL_SLOPE,
    EQUILIBRIUM_TEMPERATURE,
    GAMMA,
    VELOCITY,
    domain,
    pyclaw,
    shock_tube_initialize_state,
    solver,
):
    state = pyclaw.State(domain, solver.num_eqn)
    # Adiabatic to start with
    # _gamma = 5.0 / 3.0
    _gamma = GAMMA.value
    state.problem_data["gamma"] = _gamma
    state.problem_data["gamma1"] = _gamma - 1.0
    state.problem_data["efix"] = False

    # Parameters for the cooling
    state.problem_data["cool_method"] = COOL_METHOD.value
    state.problem_data["cool_rate"] = COOL_RATE.value
    state.problem_data["cool_slope"] = COOL_SLOPE.value
    state.problem_data["T_eq"] = EQUILIBRIUM_TEMPERATURE.value


    # Use oppositely directed velocities
    shock_tube_initialize_state(state, v_l=VELOCITY.value, v_r=-VELOCITY.value)
    state
    return (state,)


@app.cell
def __(mo):
    mo.md(r"""We will also write a function here to convert conserved variables to primitives, which we will use later in the plotting""")
    return


@app.cell
def __():
    varnames = [
        "density",
        "velocity",
        "pressure",
        "temperature",
        "sound speed",
        "mach number",
    ]
    return (varnames,)


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


    def get_variables_dict(state):
        "Return a dict of all primitive and derived variables"
        density = state.q[i_density, :]
        velocity = state.q[i_momentum, :] / density
        pressure = state.problem_data["gamma1"] * (
            state.q[i_energy, :] - 0.5 * density * velocity**2
        )
        temperature = pressure / density
        sound_speed = np.sqrt(state.problem_data["gamma"] * pressure / density)
        mach_number = np.abs(velocity / sound_speed)
        return {
            "density": density,
            "velocity": velocity,
            "pressure": pressure,
            "temperature": temperature,
            "sound speed": sound_speed,
            "mach number": mach_number,
        }
    return get_variables_dict, primitives, sound_speed


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### The controller""")
    return


@app.cell
def __(domain, pyclaw, solver, state):
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.solver = solver
    controller.tfinal = 1.0
    controller.num_output_times = 100

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
def __(controller, mo):
    ITIME = mo.ui.number(
        start=0,
        stop=controller.num_output_times,
        step=1,
        value=0,
        label="Timestep",
        debounce=True,
    )
    return (ITIME,)


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
    VELOCITY = mo.ui.number(
        start=0.01,
        stop=10.0,
        step=0.01,
        value=1.0,
        label=r"Inflow velocity",
    )
    COOL_RATE = mo.ui.number(
        start=0.0,
        stop=100.0,
        step=0.1,
        value=1.0,
        label=r"Cooling rate",
    )
    EQUILIBRIUM_TEMPERATURE = mo.ui.number(
        start=0.1,
        stop=10.0,
        step=0.1,
        value=1.0,
        label=r"Equilibrium Temperature",
    )
    COOL_SLOPE = mo.ui.number(
        start=-0.5,
        stop=5.0,
        step=0.1,
        value=2.3,
        label=r"Cooling steepness",
    )
    COOL_METHOD = mo.ui.dropdown(
        options=["first order", "second order", "exact"],
        value="second order",
        label="Cooling method",
    )
    NCELLS = mo.ui.number(
        start=100,
        stop=3200,
        step=100,
        value=800,
        label=r"Number of cells",
    )
    run_parameters = [
        NCELLS,
        GAMMA,
        VELOCITY,
        COOL_METHOD,
        COOL_RATE,
        COOL_SLOPE,
        EQUILIBRIUM_TEMPERATURE,
    ]
    return (
        COOL_METHOD,
        COOL_RATE,
        COOL_SLOPE,
        EQUILIBRIUM_TEMPERATURE,
        GAMMA,
        NCELLS,
        VELOCITY,
        run_parameters,
    )


@app.cell
def __(ITIME, NCELLS, controller, np, pd, primitives):
    _state = controller.frames[ITIME.value].state
    _d, _v, _p = primitives(_state)
    _x = _state.grid.x.centers


    def tidy(col, step=NCELLS.value // 20, precision=3):
        return np.round(col[slice(None, None, step)], precision)


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
def __(
    ITIME,
    controller,
    get_variables_dict,
    mo,
    plt,
    run_parameters,
    sns,
):
    # get the state at the current time
    _this_state = controller.frames[ITIME.value].state
    _x = _this_state.grid.x.centers
    _dependent_variables = get_variables_dict(_this_state)

    # Plot all the variables
    _nplots = len(_dependent_variables)
    _fig, _axes = plt.subplots(_nplots, 1, sharex=True)
    _colors = sns.color_palette(n_colors=_nplots)
    for _ax, _varname, _color in zip(_axes, _dependent_variables, _colors):
        _ax.plot(_x, _varname, data=_dependent_variables, color=_color)
        _ax.axvline(0.0, color="k", ls="dotted")
        _ax.legend(loc="lower left", ncol=2)
        _ax.set(
            ylim=[None, None],
        )
    _axes[-1].set_xlabel("Position, $x$")
    sns.despine(_fig)

    # Show figure by the side of parameter picker UI elements
    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    ITIME,
                    mo.md(f"Time = {_this_state.t:.3f}"),
                    run_parameters
                ]
            ),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md("""## Two-dimensional space-time arrays""")
    return


@app.cell
def __(NCELLS, controller, get_variables_dict, np, varnames):
    _nt, _nx = controller.num_output_times, NCELLS.value
    grids = {_varname: np.empty((_nt, _nx)) for _varname in varnames}
    for _i in range(_nt):
        _state = controller.frames[_i].state
        _variables = get_variables_dict(_state)
        for _varname in varnames:
            grids[_varname][_i, :] = _variables[_varname]
    grids
    return (grids,)


@app.cell
def __(matplotlib, mo, varnames):
    VARIABLE = mo.ui.dropdown(
        options=varnames, value="density", label="Dependent variable:"
    )
    COLORMAP = mo.ui.dropdown(
        options=list(matplotlib.colormaps), value="viridis", label="Color map:"
    )
    return COLORMAP, VARIABLE


@app.cell
def __(COLORMAP, VARIABLE, grids, mo, plt, run_parameters):
    _fig, _ax = plt.subplots()
    _im = _ax.imshow(
        grids[VARIABLE.value], 
        origin="lower", aspect="auto", extent=[-1, 1, 0, 1], cmap=COLORMAP.value,
    )
    _fig.colorbar(_im)
    _ax.set_title(VARIABLE.value)
    _ax.set_xlabel("Position, $x$")
    _ax.set_ylabel("Time, $t$")
    mo.hstack([_fig, mo.vstack([VARIABLE, COLORMAP, mo.md("**Run parameters:**").center(), run_parameters])])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Isothermal Mach number of the fully-radiative shock

        For cases where we cool right down to the equilibrium temperature after the shock, with zero velocity, then the total velocity change across the isothermal shock is equal to the inflow velocity, and sould be equal to the isothermal sound speed (1 in code units) times $M_0 - M_0^{-1}$, where $M_0$ is the isothermal Mach number of the shock
        """
    )
    return


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
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    import cooling_function
    return (
        cooling_function,
        i_density,
        i_energy,
        i_momentum,
        matplotlib,
        np,
        pd,
        plot,
        plt,
        pyclaw,
        riemann,
        sns,
    )


@app.cell
def __(cooling_function):
    help(cooling_function)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
