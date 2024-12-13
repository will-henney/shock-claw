import marimo

__generated_with = "0.9.33"
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


@app.cell
def __(mo):
    mo.md("""## Components of the Clawpack simulation""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### The Solver

        Use the python version of the Roe solver for now. And the boundary conditions seem to be outflow. I am taking the ones from the Sod shock problem.
        """
    )
    return


@app.cell
def __(cooling_function, domain, pyclaw, riemann):
    _ = domain  # re-create the solver if the domain should ever change
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
        ### The Domain

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
        ### The State and its initial conditions

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
    domain,
    pyclaw,
    shock,
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


    # Use oppositely directed velocities and aim for a post-cooled density of 1
    # shock_tube_initialize_state(state, v_l=VELOCITY.value, v_r=-VELOCITY.value)
    shock_tube_initialize_state(
        state,
        rho_l=1 / shock.R_2,
        p_l=1 / shock.R_2,
        v_l=shock.U_jump,
        rho_r=1 / shock.R_2,
        p_r=1 / shock.R_2,
        v_r=-shock.U_jump,
    )
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
    mo.md(
        r"""
        ### The Controller and the Solution

        The Controler is what takes charge of running the simulation. It contains a Solution, which is intialised from the State and the Domain
        """
    )
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
        value=100,
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
        stop=1000.0,
        step=0.1,
        value=10.0,
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
        step=0.01,
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
    return (
        COOL_METHOD,
        COOL_RATE,
        COOL_SLOPE,
        EQUILIBRIUM_TEMPERATURE,
        GAMMA,
        NCELLS,
        VELOCITY,
    )


@app.cell
def __(
    COOL_METHOD,
    COOL_RATE,
    COOL_SLOPE,
    EQUILIBRIUM_TEMPERATURE,
    GAMMA,
    MACH,
    NCELLS,
):
    run_parameters = [
        NCELLS,
        GAMMA,
        # VELOCITY,
        MACH,
        COOL_METHOD,
        COOL_RATE,
        COOL_SLOPE,
        EQUILIBRIUM_TEMPERATURE,
    ]
    return (run_parameters,)


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
    shock,
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
                    run_parameters,
                    vars(shock),
                ]
            ),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        """

                                    ## Two-dimensional space-time arrays
        """
    )
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
def __(COLORMAP, VARIABLE, grids, mo, plt, run_parameters, shock):
    _fig, _ax = plt.subplots()
    _im = _ax.imshow(
        grids[VARIABLE.value],
        origin="lower",
        aspect="auto",
        extent=[-1, 1, 0, 1],
        cmap=COLORMAP.value,
    )
    _fig.colorbar(_im)
    _ax.set_title(VARIABLE.value)
    _ax.set_xlabel("Position, $x$")
    _ax.set_ylabel("Time, $t$")
    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    VARIABLE,
                    COLORMAP,
                    mo.md("**Run parameters:**").center(),
                    run_parameters,
                    mo.md("**Shock conditions:**").center(),
                    vars(shock),
                ]
            ),
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Isothermal Mach number of the fully-radiative shock

        For cases where we cool right down to the equilibrium temperature after the shock, with zero velocity (from symmetry at the midplane), then the total velocity change across the isothermal shock is equal to the inflow velocity, $U$.  This should be equal to the isothermal sound speed (1 in code units) times $M_0 - M_0^{-1}$, where $M_0$ is the isothermal Mach number of the shock, so that

        \[
            M_0 = 0.5 \left(U + (U^2 + 4)^{1/2}\right) 
            = \left[1 + \left(\frac{U}{2}\right)^{\!\!2}\right]^{1/2} + \frac{U}{2}
        \]
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Normalise to the post-cooled state

        In order to check that the cooling lengths come out roughly constant, we need to do the teleological normalization to the final post-cooled density, \(\rho_2\)
        """
    )
    return


@app.cell
def __(np):
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

            assert np.isclose(self.p_1, self.R_1 * self.T_1)
    return (ShockConditions,)


@app.cell
def __(mo):
    mo.md(r"""Set up a ui element `MACH` to specify the shock strength and a global variable `shock` to hold the corresptonding data, which we can then use for the initia conditions of the simulation.""")
    return


@app.cell
def __(MACH, ShockConditions):
    # shock = ShockConditions(1.01*np.sqrt(5/3))
    shock = ShockConditions(MACH.value)
    vars(shock)
    return (shock,)


@app.cell
def __(mo):
    MACH = mo.ui.number(
        start=1.01,
        stop=10.0,
        step=0.01,
        value=2.0,
        label=r"Shock isothermal Mach number",
    )
    return (MACH,)


@app.cell
def __(MACH):
    MACH
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Calculate histograms of temperature

        We weight by the density squared times the spatial cell size, so as to get the differential emission measure. Then we divide by the bin width get the right normalization, with units of EM per Temperature interval. *Now I also mutiply by the total range `Tmax - Tmin` so that the wings are at the same height for the different shock strengths.* So the units are now EM per fraction of T interval. 
        """
    )
    return


@app.cell
def __(get_variables_dict, np, pyclaw):
    def get_DEM(
        state: pyclaw.State,
        Tlim: tuple[float | None, float | None] = (None, None),
        nbins: int = 50,
    ):
        """Get the differential emission measure of `state`"""
        variables = get_variables_dict(state)
        rho = variables["density"]
        T = variables["temperature"]
        dx = state.patch.x.delta
        Tmin, Tmax = Tlim
        if Tmin is None:
            Tmin = np.min(T)
        if Tmax is None:
            Tmax = np.max(T)
        weights = dx * rho**2
        H, edges = np.histogram(
            T, weights=weights, density=False, bins=nbins, range=(Tmin, Tmax)
        )
        # Divide by bin widths
        bin_widths = edges[1:] - edges[0:-1]
        H /= bin_widths
        # Calculate bin centers
        Tgrid = (edges[:-1] + edges[1:]) / 2
        # Check that the normalization is correct (this will fail if [Tmin, Tmax] does not include all cells)
        # integral_dT = np.trapezoid(H, Tgrid) # TRAPEZOID VERSION DOES NOT WORK
        # integral_dx = np.trapezoid(rho ** 2, state.patch.x.centers)
        integral_dT = np.sum(H * bin_widths)
        integral_dx = np.sum(weights)
        assert (
            np.isclose(integral_dT, integral_dx, rtol=0.01),
            f"Mismatched integrals at time {state.t:.2f}: {integral_dT:.3f} {integral_dx:.3f}",
        )
        return {"T": Tgrid, "DEM": H * (Tmax - Tmin), "edges": edges}
    return (get_DEM,)


@app.cell
def __(ITIME, controller, get_DEM):
    _this_state = controller.frames[ITIME.value].state
    get_DEM(_this_state)
    return


@app.cell
def __(ITIME, controller):
    current_state = controller.frames[ITIME.value].state
    current_state.patch.x.centers
    return (current_state,)


@app.cell
def __(domain):
    cell_size = domain.grid.dimensions[0].delta
    return (cell_size,)


@app.cell
def __(mo):
    mo.md(r"""Parameters for the DEM plot. We wrap it in a form so that we can set all the parameters and then replot.""")
    return


@app.cell
def __(matplotlib, mo):
    DEM_PARAMS = mo.md("""
        **Histogram parameters.**

        {palette}

        {xscale} {yscale}

        {nbins} 
        
        {exponent} 
        
        {height} 
        
        {center}
        """).batch(
        palette=mo.ui.dropdown(
            options=list(matplotlib.colormaps),
            value="turbo",
            label="Color palette:",
        ),
        xscale=mo.ui.dropdown(
            options=["linear", "log"], value="linear", label="x-axis scale:"
        ),
        yscale=mo.ui.dropdown(
            options=["linear", "log"], value="log", label="y-axis scale:"
        ),
        nbins=mo.ui.number(start=10, stop=100, value=50, label="\# of bins"),
        exponent=mo.ui.number(
            start=1, stop=4, value=2.5, step=0.1, label="power-law slope"
        ),
        height=mo.ui.number(
            start=0.01, stop=10.0, value=0.5, step=0.01, label="power-law height"
        ),
        center=mo.ui.number(
            start=0.5, stop=1.5, value=1.0, step=0.01, label="power-law center"
        ),
    )

    DEM_PARAMS_FORM = DEM_PARAMS.form(
        show_clear_button=True,
        bordered=True,
        submit_button_label="Replot",
        clear_button_label="Reset",
        clear_button_tooltip="Reset all parameters to default values",
    )
    return DEM_PARAMS, DEM_PARAMS_FORM


@app.cell
def __(DEM_PARAMS_FORM):
    DEM_PARAMS_FORM.value
    return


@app.cell
def __(
    DEM_PARAMS,
    DEM_PARAMS_FORM,
    MACH,
    controller,
    get_DEM,
    mo,
    np,
    plt,
    shock,
    sns,
):
    if DEM_PARAMS_FORM.value:
        _params = DEM_PARAMS_FORM.value
    else:
        _params = DEM_PARAMS.value

    _fig, _ax = plt.subplots()
    colors = sns.color_palette(_params["palette"], n_colors=len(controller.frames))
    for _i, _frame in enumerate(controller.frames):
        _data = get_DEM(_frame.state, nbins=_params["nbins"])
        _alpha = 0.3 * np.sqrt(_i / 100)
        _ax.plot(
            "T",
            "DEM",
            data=_data,
            color=colors[_i],
            alpha=_alpha,
            lw=0.5,
            # drawstyle="steps-mid",
        )
    _vline_style = dict(color="k", lw=1, ls="dotted")
    _ax.axvline(1.0, **_vline_style)
    _ax.axvline(shock.T_1, **_vline_style)
    # Plot limits
    xmin, xmax = 0.9, 1.1 * shock.T_1
    if _params["yscale"] == "log":
        ymin, ymax = 2e-3, 30.0
    else:
        ymin, ymax = 0.0, 2.0

    # Show power-law slope
    xgrid = np.geomspace(xmin, xmax)
    _q, _A, _x0 = -_params["exponent"], _params["height"], _params["center"]
    ygrid = np.where(
        xgrid > _x0,
        _A * (xgrid - _x0) ** _q,
        np.nan,
    )
    _ax.plot(xgrid, ygrid, linestyle="dashed", color="k", alpha=0.5)

    _ax.set(
        xscale=_params["xscale"],
        yscale=_params["yscale"],
        xlabel="Temperature: $T / T_0$",
        ylabel="DEM",
        xlim=[xmin, xmax],
        ylim=[ymin, ymax],
    )
    _ax.set_title("Differential Emission Measure")
    sns.despine()
    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    DEM_PARAMS_FORM,
                    mo.md(f"""
                **Shock parameters:**

                {mo.as_html(vars(shock))}
                """),
                    MACH,
                ],
                align="center",
                gap=2,
            ),
        ],
        justify="center",
    )
    return colors, xgrid, xmax, xmin, ygrid, ymax, ymin


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Imports

        Put them all at the endo of the notebooto keep them out of the way
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
