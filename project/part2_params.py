"""
Part 2: change parameters of the OSDR ODE and see how the immune pulse changes.

    python project/part2_params.py                 # all experiments
    python project/part2_params.py --exp x0        # only one: x0 | death | binhib | truncate
    python project/part2_params.py --show

Experiments:
    x0        initial number of CD4 T cells -> excitability threshold
    death     constant death rates p- scaled by 0.5 ... 1.5   (paper: Supplementary Fig. 4e)
    binhib    strength of every B-containing term in the T cell equations scaled by 0 ... 2
    truncate  division-rate truncation at p_max on / off

Figures are saved under project/figures/.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from osdr_ode import (
    ESCAPE_CELLS,
    FIG4F_T_END,
    FIG4F_X0,
    division_rates,
    escape_time,
    load_params,
    simulate,
    with_b_inhibition_scale,
    with_death_scale,
    with_truncation,
)
from plotting import FIGURE_DIR, save

T = np.linspace(0, FIG4F_T_END, 5001)


def pulse_stats(sol):
    """Peak height, peak time and final value of each cell type (NaN-aware for escaped trajectories)."""
    t_esc = escape_time(sol, T)
    if t_esc is not None:
        nan = np.full(sol.shape[1], np.nan)
        return {"peak": nan, "t_peak": nan, "end": nan, "escape": t_esc}
    k = sol.argmax(axis=0)
    return {"peak": sol.max(axis=0), "t_peak": T[k], "end": sol[-1], "escape": None}


def print_table(name, values, stats, cell_types):
    header = f"{name:>10} | " + " | ".join(f"{c:>3} peak (t)      end" for c in cell_types)
    print(header)
    print("-" * len(header))
    for v, s in zip(values, stats):
        if s["escape"] is not None:
            print(f"{v:>10g} | escaped: some cell type exceeded {ESCAPE_CELLS:g} cells at t={s['escape']:.0f}")
            continue
        cols = [f"{s['peak'][i]:6.1f} ({s['t_peak'][i]:4.0f}) {s['end'][i]:7.2f}" for i in range(len(cell_types))]
        print(f"{v:>10g} | " + " | ".join(cols))
    print()


def plot_sweep(name, values, sols, stats, cell_types, title, value_label, t_max=2500, log_x=False):
    """Row 1: time series of each cell type, one line per parameter value. Row 2: peak height / time."""
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(values) - 1)) for i in range(len(values))]
    mask = T <= t_max

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, c in enumerate(cell_types):
        ax = axs[0, i]
        for v, sol, col in zip(values, sols, colors):
            ax.plot(T[mask], sol[mask, i], color=col, lw=1.8, label=f"{v:g}")
        ax.set_title(c)
        ax.set_xlabel("t (steps)")
        ax.set_ylabel("Number of cells")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 2].legend(title=value_label, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    peaks = np.array([s["peak"] for s in stats])
    t_peaks = np.array([s["t_peak"] for s in stats])
    ends = np.array([s["end"] for s in stats])
    for ax, data, ylabel in zip(axs[1], [peaks, t_peaks, ends], ["peak height", "time of peak", "final value (t=5000)"]):
        for i, c in enumerate(cell_types):
            ax.plot(values, data[:, i], "o-", label=c, color=plt.get_cmap("tab10")(i))
        ax.set_xlabel(value_label)
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axs[1, 0].legend(frameon=False)

    fig.suptitle(title)
    save(fig, f"part2_{name}.png")


def run_sweep(name, values, make, title, value_label, cell_types):
    sols, stats = [], []
    for v in values:
        params, x0 = make(v)
        sol = simulate(params, x0, T)
        sols.append(sol)
        stats.append(pulse_stats(sol))
    print(f"===== {title}")
    print_table(value_label, values, stats, cell_types)
    plot_sweep(name, values, sols, stats, cell_types, title, value_label)
    return sols, stats


def exp_x0(params):
    """Excitability: below a threshold of initial CD4 cells no pulse is generated."""
    values = [2, 5, 8, 10, 12, 15, 20, 30, 40]
    run_sweep(
        "x0",
        values,
        lambda v: (params, np.array([v, FIG4F_X0[1], FIG4F_X0[2]])),
        "Initial CD4 T cells (CD8 = 4, B = 1)",
        "CD4(0)",
        params.cell_types,
    )

    # finer scan: threshold as the CD4(0) at which CD4 starts to grow instead of decaying
    fine = np.linspace(1, 40, 79)
    growth = np.array([simulate(params, [v, FIG4F_X0[1], FIG4F_X0[2]], T)[:, 0].max() - v for v in fine])
    threshold = fine[np.argmax(growth > 1e-6)] if np.any(growth > 1e-6) else np.nan
    print(f"excitability threshold: CD4 grows after t=0 only if CD4(0) >= ~{threshold:.1f}\n")


def exp_death(params):
    values = [0.5, 0.75, 0.85, 0.9, 1.0, 1.25, 1.5]
    run_sweep(
        "death",
        values,
        lambda v: (with_death_scale(params, v), FIG4F_X0),
        "Death rate p- scaled (x0 = Fig. 4f)",
        "p- scale",
        params.cell_types,
    )


def exp_binhib(params):
    values = [0.0, 0.5, 1.0, 1.5, 2.0]
    run_sweep(
        "binhib",
        values,
        lambda v: (with_b_inhibition_scale(params, v), FIG4F_X0),
        "B-cell terms in T cell equations scaled (x0 = Fig. 4f)",
        "B->T scale",
        params.cell_types,
    )


def exp_truncate(params):
    values = [1.0, 0.0]  # 1 = truncation on (fitted model), 0 = off
    run_sweep(
        "truncate",
        values,
        lambda v: (with_truncation(params, bool(v)), FIG4F_X0),
        "Division-rate truncation at p_max (1 = on, 0 = off)",
        "truncate",
        params.cell_types,
    )

    # the truncation only matters where sigmoid(z) > p_max:
    sol = simulate(with_truncation(params, False), FIG4F_X0, T)
    max_rate = division_rates(with_truncation(params, False), sol).max(axis=0)
    for c, r, pm in zip(params.cell_types, max_rate, params.p_max):
        print(f"  {c}: max division rate along the Fig. 4f trajectory {r:.4f}  vs  p_max {pm:.4f}")
    print()


EXPERIMENTS = {"x0": exp_x0, "death": exp_death, "binhib": exp_binhib, "truncate": exp_truncate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", choices=list(EXPERIMENTS), action="append", help="run only these experiments")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args()

    params = load_params()
    for name in args.exp or list(EXPERIMENTS):
        EXPERIMENTS[name](params)

    print(f"figures saved to {FIGURE_DIR}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
