import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""This will include material copied from the [web page of the PyClaw tutorial](http://www.clawpack.org/pyclaw/tutorial.html)""")
    return


@app.cell
def __(mo):
    mo.md("""## The Solver""")
    return


@app.cell
def __(pyclaw, riemann):
    solver = pyclaw.ClawSolver1D(riemann.acoustics_1D)
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver
    return (solver,)


@app.cell
def __(mo):
    mo.md(r"""## The domain""")
    return


@app.cell
def __(pyclaw):
    domain = pyclaw.Domain([-1.0], [1.0], [200])
    return (domain,)


@app.cell
def __(domain, pyclaw, solver):
    solution = pyclaw.Solution(solver.num_eqn, domain)
    return (solution,)


@app.cell
def __(mo):
    mo.md(r"""## Initial Condition""")
    return


@app.cell
def __(np, solution):
    state = solution.state
    xc = state.grid.p_centers[0]      # Array containing the cell center coordinates
    from numpy import exp
    state.q[0,:] = exp(-100 * (xc-0.25)**2) # Pressure: Gaussian centered at x=0.75.
    state.q[1,:] = 0.                       # Velocity: zero.

    # Problem-specific parameters
    rho = 1.0
    bulk = 1.0
    state.problem_data['rho'] = rho
    state.problem_data['bulk'] = bulk
    state.problem_data['zz'] = np.sqrt(rho*bulk)
    state.problem_data['cc'] = np.sqrt(bulk/rho)
    return bulk, exp, rho, state, xc


@app.cell
def __(mo):
    mo.md(r"""### The controller""")
    return


@app.cell
def __(pyclaw, solution, solver):
    controller = pyclaw.Controller()
    controller.solution = solution
    controller.solver = solver
    controller.tfinal = 5.0
    controller.num_output_times = 50

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
def __(controller):
    len(controller.out_times)
    return


@app.cell
def __(controller):
    controller.frames[22].state.t
    return


@app.cell
def __(controller, itime, np):
    np.round(
        controller.frames[itime.value].state.q[:, ::20],
        3
    )
    return


@app.cell
def __(controller, domain, itime, mo, plt):
    fig, ax = plt.subplots()
    _x = domain.grid.c_centers[0]
    _p = controller.frames[itime.value].state.q[0, ...]
    _v = controller.frames[itime.value].state.q[1, ...]
    ax.plot(_x, _p, label="pressure")
    ax.plot(_x, _v, label="velocity")
    ax.axvline(-1.0, color="k")
    ax.axvline(1.0, color="k")
    ax.legend(loc="lower left")
    ax.set(
        ylim=[-1, 1],
    )
    mo.hstack(
        [
            fig, 
            mo.vstack([
                itime,
                mo.md(f"Time = {controller.frames[itime.value].state.t:.1f}"),
            ]),
        ]
    )
    return ax, fig


@app.cell
def __(controller, mo):
    itime = mo.ui.number(
        start=0, stop=controller.num_output_times, step=1,
        label="Timestep", debounce=True,
    )
    return (itime,)


@app.cell
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
    from clawpack.pyclaw import plot
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return np, plot, plt, pyclaw, riemann, sns


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
