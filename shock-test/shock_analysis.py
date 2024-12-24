import typer
from clawpack import pyclaw
from clawpack.riemann.euler_with_efix_1D_constants import (
    density as i_density,
    momentum as i_momentum,
    energy as i_energy,
)
from pathlib import Path
import shock_utils
from matplotlib import pyplot as plt
import seaborn as sns


def load_frames(model_path: Path, max_frames=1000) -> list[pyclaw.Solution]:
    """Read all output time frames from a simulation."""
    frames = []
    for iframe in range(max_frames + 1):
        solution = pyclaw.Solution()
        try:
            solution.read(
                iframe, path=str(model_path), file_format="hdf5", file_prefix=None
            )
        except FileNotFoundError:
            # Stop when we run out of files
            break
        frames.append(solution)
    return frames


def make_line_plots(
    model_path: Path,
    frames: list[pyclaw.Solution],
    plot_dir: Path = Path("shock-plots"),
    plot_vars: list[str] | None = None,
) -> None:
    """Make line plots of the solution variables.

    Optionally specify a list of variables to plot with `plot_vars`. If not specified, all
    """
    plot_dir.mkdir(exist_ok=True)
    model_prefix = model_path.stem
    first_state = frames[0].state
    x = first_state.grid.x.centers
    plot_vars = plot_vars or shock_utils.get_variables_dict(first_state).keys()
    nplots = len(plot_vars)
    fig, axes = plt.subplots(nplots, 1, sharex=True, figsize=(8, 2 * nplots))
    colors = sns.color_palette(n_colors=nplots)
    for iframe, frame in enumerate(frames):
        data = shock_utils.get_variables_dict(frame.state)
        for ax, varname, color in zip(axes, plot_vars, colors):
            ax.plot(x, varname, data=data, color=color, alpha=iframe/len(frames))
    for ax, varname in zip(axes, plot_vars):
        ax.axvline(0.0, color="k", linestyle="dotted")
        ax.set_ylabel(varname)
    axes[-1].set_xlabel("Position, $x$")
    sns.despine(fig)
    fig.savefig(plot_dir / f"lineplot-{model_prefix}.pdf")


def main(output_dir: Path = Path("shock-output")):
    model_paths = output_dir.glob("shock-Ma-*N-????")
    for model_path in model_paths:
        frames = load_frames(model_path)
        make_line_plots(model_path, frames[::10])

if __name__ == "__main__":
    typer.run(main)
