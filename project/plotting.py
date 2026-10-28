"""
Shared plotting helpers for the project/ scripts.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parent / "figures"

PROJECTIONS = [(0, 1), (0, 2), (1, 2)]  # (CD4, CD8), (CD4, B), (CD8, B)


def save(fig, name: str):
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=200, bbox_inches="tight")


def plot_timeseries(ax, t, sol, cell_types, colors, linestyle="-", label_suffix="", alpha=1.0, linewidth=2.0):
    for i, c in enumerate(cell_types):
        ax.plot(t, sol[:, i], color=colors[c], linestyle=linestyle, label=f"{c}{label_suffix}", alpha=alpha, lw=linewidth)
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("Number of cells")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def plot_3d_trajectory(
    sol, cell_types, title: str, color="#3B76BC", plot_projections=True, lims=None, invert_y: bool = False
):
    """Same style as tdm.plot.three_cells.trajectory.plot_3D_trajectory (linear space), for any trajectory.

    invert_y=True flips the second axis (CD8) so that its zero lies next to the vertical (B) axis.
    """
    xs, ys, zs = sol[:, 0], sol[:, 1], sol[:, 2]
    lims = lims or [(0, max(1.0, np.nanmax(sol[:, i]) * 1.05)) for i in range(3)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.plot(xs, ys, zs, color=color, linewidth=3, label="Trajectory")

    if plot_projections:
        gray = dict(color="gray", alpha=0.5, linewidth=3)
        y_back = lims[1][0] if invert_y else lims[1][1]  # the CD8 wall at the back of the box
        ax.plot(xs, ys, np.full_like(zs, lims[2][0]), **gray)
        ax.plot(xs, np.full_like(ys, y_back), zs, **gray)
        ax.plot(np.full_like(xs, lims[0][0]), ys, zs, **gray)

    ax.scatter(*sol[0], color="#6BBF59", s=60, label="Start state", depthshade=False)
    ax.scatter(*sol[-1], color="#E4572E", s=60, label="Final state", depthshade=False)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*(lims[1][::-1] if invert_y else lims[1]))
    ax.set_zlim(*lims[2])
    ax.set_xlabel(cell_types[0])
    ax.set_ylabel(cell_types[1])
    ax.set_zlabel(cell_types[2])
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_3d_overlay(trajectories, cell_types, title: str, plot_projections=True, invert_y: bool = False):
    """Several trajectories in one 3D box, each with its own color / line style and wall projections.

    trajectories: list of (sol, label, color, linestyle); all share the start state of the first one.
    """
    sols = [t[0] for t in trajectories]
    lims = [(0, max(1.0, max(np.nanmax(s[:, i]) for s in sols) * 1.05)) for i in range(3)]
    y_back = lims[1][0] if invert_y else lims[1][1]  # the CD8 wall at the back of the box

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    for sol, label, color, ls in trajectories:
        xs, ys, zs = sol[:, 0], sol[:, 1], sol[:, 2]
        ax.plot(xs, ys, zs, color=color, linewidth=3, linestyle=ls, label=label)
        if plot_projections:
            proj = dict(color=color, alpha=0.3, linewidth=2.5, linestyle=ls)
            ax.plot(xs, ys, np.full_like(zs, lims[2][0]), **proj)
            ax.plot(xs, np.full_like(ys, y_back), zs, **proj)
            ax.plot(np.full_like(xs, lims[0][0]), ys, zs, **proj)

    start = sols[0][0]
    ax.scatter(*start, color="#6BBF59", s=70, label="Start state", depthshade=False, zorder=10)
    for sol, label, color, _ in trajectories:
        end = sol[~np.isnan(sol).any(axis=1)][-1]
        ax.scatter(*end, color=color, s=60, marker="X", label=f"Final state ({label})", depthshade=False, zorder=10)

    ax.set_xlim(*lims[0])
    ax.set_ylim(*(lims[1][::-1] if invert_y else lims[1]))
    ax.set_zlim(*lims[2])
    ax.set_xlabel(cell_types[0])
    ax.set_ylabel(cell_types[1])
    ax.set_zlabel(cell_types[2])
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    return fig
