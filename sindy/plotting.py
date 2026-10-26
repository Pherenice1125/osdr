"""
Shared plotting helpers for the sindy/ scripts.
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


def plot_3d_trajectory(sol, cell_types, title: str, color="#3B76BC", plot_projections=True, lims=None):
    """Same style as tdm.plot.three_cells.trajectory.plot_3D_trajectory (linear space), for any trajectory."""
    xs, ys, zs = sol[:, 0], sol[:, 1], sol[:, 2]
    lims = lims or [(0, max(1.0, sol[:, i].max() * 1.05)) for i in range(3)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.plot(xs, ys, zs, color=color, linewidth=3, label="Trajectory")

    if plot_projections:
        gray = dict(color="gray", alpha=0.5, linewidth=3)
        ax.plot(xs, ys, np.full_like(zs, lims[2][0]), **gray)
        ax.plot(xs, np.full_like(ys, lims[1][1]), zs, **gray)
        ax.plot(np.full_like(xs, lims[0][0]), ys, zs, **gray)

    ax.scatter(*sol[0], color="#6BBF59", s=60, label="Start state", depthshade=False)
    ax.scatter(*sol[-1], color="#E4572E", s=60, label="Final state", depthshade=False)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_xlabel(cell_types[0])
    ax.set_ylabel(cell_types[1])
    ax.set_zlabel(cell_types[2])
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    return fig
