"""
Part 1: the OSDR ODE behind Fig. 4f.

    python project/part1_timeseries.py            # print the 3 ODEs, plot time series + 3D trajectory
    python project/part1_timeseries.py --verify   # also check osdr_ode.rhs against the original tdm model
    python project/part1_timeseries.py --show     # open the figures in a window (they are always saved)

Figures are saved under project/figures/.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from osdr_ode import (
    CELL_COLORS,
    FIG4F_T_END,
    FIG4F_X0,
    format_equations,
    load_params,
    rhs,
    simulate,
    sympy_equations,
)
from plotting import FIGURE_DIR, plot_3d_trajectory, plot_timeseries, save

T_END = 1500.0  # the pulse is over by t ~ 2000 (Fig. 4f itself integrates to 5000)


def verify_against_tdm(params, n_states: int = 2000, seed: int = 0):
    """rhs() must agree with tdm's model.delta_cells and compute_trajectory (the code that draws Fig. 4f)."""
    import pandas as pd
    from tdm.analysis import Analysis
    from tdm.simulate.deterministic import compute_trajectory
    from osdr_ode import ANALYSIS_CACHE_NAME, load_data_states

    print("verifying against the tdm model (loads the cached analysis)...")
    ana = Analysis.load(ANALYSIS_CACHE_NAME)

    # 1. vector field at observed states and at random states (incl. outside the data):
    rng = np.random.default_rng(seed)
    X = np.vstack([load_data_states()[:n_states], rng.uniform(0, 120, size=(n_states, 3))])
    tdm_f = np.array(ana.model.delta_cells(pd.DataFrame(X, columns=ana.cell_types), return_order=ana.cell_types)).T
    err = np.abs(rhs(params, X) - tdm_f).max()
    print(f"  max |rhs - tdm.delta_cells| over {len(X)} states: {err:.2e}")

    # 2. the Fig. 4f trajectory itself:
    t = np.linspace(0, FIG4F_T_END, int(FIG4F_T_END) + 1)
    sol_tdm = compute_trajectory(ana, initial_cell_counts=list(FIG4F_X0), odeint_timepoints=t)
    sol_ours = simulate(params, FIG4F_X0, t)
    print(f"  max |trajectory - tdm.compute_trajectory| (cells): {np.abs(sol_ours - sol_tdm).max():.2e}")
    print("  (tdm integrates with atol=1e-3, so differences of ~1e-3 cells are integration error)\n")


def describe_pulse(params, sol, t):
    for i, c in enumerate(params.cell_types):
        k = sol[:, i].argmax()
        print(f"  {c:>3}: start {sol[0, i]:6.2f}  peak {sol[k, i]:6.2f} at t={t[k]:6.0f}  end {sol[-1, i]:6.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="compare with the original tdm model")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args()

    params = load_params()

    print("=" * 30, "OSDR ODE (Fig. 4f model)", "=" * 30)
    print(format_equations(params), "\n")
    print("sympy form (sigma(z) = 1 / (1 + exp(-z))):")
    for ode, z in sympy_equations(params):
        sp.pprint(ode, wrap_line=False, use_unicode=True)
        sp.pprint(z, wrap_line=False, use_unicode=True)
        print()
    print("LaTeX:")
    for ode, z in sympy_equations(params, digits=3):
        print(" ", sp.latex(ode))
        print(" ", sp.latex(z))
    print()

    if args.verify:
        verify_against_tdm(params)

    t = np.linspace(0, T_END, int(T_END) + 1)
    sol = simulate(params, FIG4F_X0, t)
    print(f"Fig. 4f trajectory from x0 = {FIG4F_X0.tolist()}:")
    describe_pulse(params, sol, t)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_timeseries(ax, t, sol, params.cell_types, CELL_COLORS)
    ax.set_title("OSDR ODE: Fig. 4f initial state (CD4, CD8, B) = (40, 4, 1)")
    save(fig, "part1_timeseries.png")

    fig = plot_3d_trajectory(sol, params.cell_types, title="OSDR (Fig. 4f)", invert_y=True)
    save(fig, "part1_trajectory_3d.png")

    print(f"figures saved to {FIGURE_DIR}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
