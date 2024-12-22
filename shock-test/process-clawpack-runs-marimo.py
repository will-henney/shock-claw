import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Test of reading back in Clawpack runs

        I have done production runs, writing results to HDF files, but I need to check that we can read them back in again.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""First we test reading in a single file""")
    return


@app.cell
def _(pyclaw):
    solution = pyclaw.Solution()
    solution.read(
        50,
        path="shock-output/shock-Ma-4.0-Lambda-10.0-q-2.3-N-3200",
        file_format="hdf5",
        file_prefix=None,
    )
    solution
    return (solution,)


@app.cell
def _(pyclaw):
    def load_frames(model_id="Ma-4.0-Lambda-10.0", nframes=100):
        data_path = f"shock-output/shock-{model_id}-q-2.3-N-3200"
        frames = []
        for iframe in range(nframes + 1):
            solution = pyclaw.Solution()
            solution.read(
                iframe, path=data_path, file_format="hdf5", file_prefix=None
            )
            frames.append(solution)
        return frames
    return (load_frames,)


@app.cell
def _(load_frames):
    load_frames()
    return


@app.cell
def _(mo):
    mo.md(r"""So that seems to work. We will now implement everything in a proper script.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    from clawpack import pyclaw
    from clawpack import riemann

    # Indices of each conserved variable in the solution arrays
    from clawpack.riemann.euler_with_efix_1D_constants import (
        density as i_density,
        momentum as i_momentum,
        energy as i_energy,
    )
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
        plt,
        pyclaw,
        riemann,
        sns,
    )


app._unparsable_cell(
    r"""
    3import marimo as mo
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
