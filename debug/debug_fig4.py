"""
Step-by-step reproduction of Fig. 4 (excitable T / B cell circuit).

Mirrors the functions in src/tdm/publications/first/figures/f4.py, but bypasses @persistent_cache
so that breakpoints inside tdm are always hit, and prints the intermediate numbers behind each panel.

    python debug/debug_fig4.py            # all panels
    python debug/debug_fig4.py --d        # only Fig. 4d
    python debug/debug_fig4.py --d --g    # Fig. 4d and 4g
    python debug/debug_fig4.py --g --refit

Panels:
    --abc  Ki67 vs. neighbourhood (a, b) + T-B phase portrait with a flare trajectory (c)   [2D T-B model]
    --d    stochastic spatial simulation of a single immune flare                          [2D T-B model]
    --e    refractory period: T cell influx at t=0, 1000, 2500                             [2D T-B model]
    --f    3D CD4-CD8-B trajectory                                                         [3D model]
    --g    neighbour counts in the 100 most / least proliferative states                   [3D model]

Run with the "Debug Fig. 4" launch configurations.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from tdm.analysis import Analysis
from tdm.cell_types import (
    FIBROBLAST,
    MACROPHAGE,
    TUMOR,
    ENDOTHELIAL,
    T_CELL,
    B_CELL,
    CD4_T_CELL,
    CD8_T_CELL,
)
from tdm.paths import CACHED_ANALYSES_DIR
from tdm.plot.model.interpret import plot_high_and_low_proliferation_states
from tdm.plot.three_cells.trajectory import plot_3D_trajectory
from tdm.plot.tissue.stepper import plot_cells_over_time
from tdm.plot.two_cells.phase_portrait import plot_growth_rate, plot_trajectory
from tdm.preprocess.single_cell_df import CELL_TYPE_COL, DIVISION_COL
from tdm.raw.breast_mibi import read_single_cell_df, CELL_TYPE_DEFINITIONS_CD4_CD8_B
from tdm.simulate.deterministic import compute_trajectory
from tdm.simulate.generate_distribution import simulate_one_tissue, get_tissue0, n_cells_per_neighborhood
from tdm.utils import microns, inv_log2_1p

SEED = 42  # f4.py uses np.random.seed(42) for 4d / 4e
K_STATES = 100  # hard-coded in plot_high_and_low_proliferation_states
TISSUE_SIZE = microns(2000)  # 2 mm x 2 mm simulated tissue (4d, 4e)

CACHE_2D = "debug_fig4_tb2d.pkl"  # saved under .cached_analyses/
CACHE_3D = "debug_fig4_tb3d.pkl"


"""
Analyses (models):
"""


def print_data_summary(df):
    print("n cells per type:\n", df[CELL_TYPE_COL].value_counts(), "\n")
    print("fraction dividing (Ki67 > threshold):\n", df.groupby(CELL_TYPE_COL)[DIVISION_COL].mean(), "\n")


def fit_2d_analysis():
    """Same arguments as tb_analysis() in publications/first/analyses.py."""
    df = read_single_cell_df()
    print_data_summary(df)

    return Analysis(
        single_cell_df=df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, T_CELL, B_CELL],  # all allowed
        cell_types_to_model=[T_CELL, B_CELL],
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": False,
            "scale_features": False,
        },
        enforce_max_density=True,
        max_density_enforcer_power=8,
        verbose=True,
    )


def fit_3d_analysis():
    """Same arguments as get_3d_tb_ana() in publications/first/analyses.py."""
    df = read_single_cell_df(cell_type_definitions=CELL_TYPE_DEFINITIONS_CD4_CD8_B)
    print_data_summary(df)

    return Analysis(
        single_cell_df=df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        cell_types_to_model=[CD4_T_CELL, CD8_T_CELL, B_CELL],
        polynomial_dataset_kwargs={
            "degree": 2,
        },
        enforce_max_density=False,
        max_density_enforcer_power=8,
        verbose=True,
    )


_analyses: dict[str, Analysis] = {}


def get_analysis(cache_name: str, fit_func, refit: bool) -> Analysis:
    """Load from .cached_analyses/ unless refit=True; reused across panels within one run."""
    if cache_name in _analyses:
        return _analyses[cache_name]

    if not refit and (CACHED_ANALYSES_DIR / cache_name).exists():
        print(f"loading cached analysis: {CACHED_ANALYSES_DIR / cache_name}\n")
        ana = Analysis.load(cache_name)
    else:
        ana = fit_func()
        ana.dump(cache_name)

    print(ana, "\n")
    _analyses[cache_name] = ana
    return ana


def get_2d_ana(refit: bool) -> Analysis:
    return get_analysis(CACHE_2D, fit_2d_analysis, refit)


def get_3d_ana(refit: bool) -> Analysis:
    return get_analysis(CACHE_3D, fit_3d_analysis, refit)


"""
Helpers:
"""


def counts_per_neighborhood(stepper, cell_type: str) -> np.ndarray:
    """Number of cells per 80um neighbourhood at every simulation step (the y axis of 4d / 4e)."""
    n_cells = np.array([(t.cell_df().cell_type == cell_type).sum() for t in stepper.tissues])
    return n_cells_per_neighborhood(
        n_cells_per_tissue=n_cells,
        tissue_width=stepper.tissue_width,
        tissue_height=stepper.tissue_height,
        neighborhood_size=stepper.neighborhood_size,
        round_to_int=False,
    )


def print_pulse_summary(stepper, cell_types: list[str]):
    for cell_type in cell_types:
        n = counts_per_neighborhood(stepper, cell_type)
        print(
            f"{cell_type}: start {n[0]:.1f}, peak {n.max():.1f} at step {n.argmax()}, end {n[-1]:.1f} "
            f"(per neighbourhood, {len(n)} steps)"
        )
    print()


def print_trajectory_summary(sol: np.ndarray, cell_types: list[str], timepoints: np.ndarray):
    for i, cell_type in enumerate(cell_types):
        peak = sol[:, i].argmax()
        print(
            f"{cell_type}: start {sol[0, i]:.2f}, peak {sol[peak, i]:.2f} at t={timepoints[peak]:.0f}, "
            f"end {sol[-1, i]:.2f}"
        )
    print()


"""
Panels:
"""


def debug_fig_4abc(refit: bool):
    """Fig. 4a-c: Ki67 over the T-B plane and the excitable phase portrait."""
    ana = get_2d_ana(refit)

    odeint_timepoints = np.linspace(0, 3500, 10000)
    arrow_timepoint_idxs = [0, 1250, 1900, 2800, 5900]
    state0 = (4.0, 0.001)  # (T, B) in log2(1 + n) units

    # deterministic flare trajectory (blue line in 4c), in absolute cell numbers:
    sol = compute_trajectory(ana, initial_cell_counts=inv_log2_1p(np.array(state0)), odeint_timepoints=odeint_timepoints)
    print("4c flare trajectory (cells per neighbourhood):")
    print_trajectory_summary(sol, ana.cell_types, odeint_timepoints)

    fig = plot_growth_rate(ana, plot_nullclines=False, include_titles=False)
    plot_trajectory(
        ana,
        state0=state0,
        odeint_timepoints=odeint_timepoints,
        arrow_timepoint_idxs=arrow_timepoint_idxs,
        ax=fig.axes[0],
    )
    fig.suptitle("Fig. 4a-c")
    plt.show()


def debug_fig_4d(refit: bool):
    """Fig. 4d: stochastic simulation of a 2 mm x 2 mm tissue -> a single immune flare."""
    np.random.seed(SEED)
    ana = get_2d_ana(refit)

    state0 = (4.5, 1.0)  # (T, B) in log2(1 + n) units per neighbourhood
    print(f"4d initial state: T={inv_log2_1p(state0[0]):.1f}, B={inv_log2_1p(state0[1]):.1f} cells per neighbourhood")

    stepper = simulate_one_tissue(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        model=ana.model,
        neighborhood_size=ana.neighborhood_size,
        n_steps=2000,
        tissue_width=TISSUE_SIZE,
        tissue_height=TISSUE_SIZE,
        state0=state0,
        return_stepper=True,
        verbose=True,
        diffusion_coef=0,
    )
    # expected (paper): T rises first (peak ~50 at step ~250), B follows (peak ~22 at step ~500), both decay
    print_pulse_summary(stepper, [T_CELL, B_CELL])

    fig, ax = plt.subplots(figsize=(5, 2))
    plot_cells_over_time(
        stepper=stepper,
        cell_types=[T_CELL, B_CELL],
        log_cells=False,
        legend_loc="upper right",
        ax=ax,
        ylim=(0, 6.2),
        n_steps=1500,
    )
    ax.set_title("Fig. 4d")
    plt.show()


def debug_fig_4e(refit: bool):
    """Fig. 4e: refractory period - T cell influx during / after a flare."""
    np.random.seed(SEED)
    ana = get_2d_ana(refit)

    def add_t_cell_influx(stepper):
        influx = get_tissue0(
            cell_a=ana.cell_a,
            cell_b=ana.cell_b,
            start_a=4.5,
            start_b=0,
            nbrhood_size=ana.neighborhood_size,
            tissue_width=TISSUE_SIZE,
            tissue_height=TISSUE_SIZE,
        )
        stepper.tissues[-1] = stepper.tissues[-1] + influx

    stepper = simulate_one_tissue(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        model=ana.model,
        neighborhood_size=ana.neighborhood_size,
        n_steps=1000,
        tissue_width=TISSUE_SIZE,
        tissue_height=TISSUE_SIZE,
        state0=(4.5, 0.5),
        return_stepper=True,
        verbose=True,
        diffusion_coef=0,
    )
    print("after first flare (t=1000):")
    print_pulse_summary(stepper, [T_CELL, B_CELL])

    # influx @ 1000: B cells still high -> no pulse expected
    add_t_cell_influx(stepper)
    stepper.step_n_times(1500, verbose=True)

    # influx @ 2500: B cells have declined -> new pulse expected
    add_t_cell_influx(stepper)
    stepper.step_n_times(1000, verbose=True)

    print("full run (t=0..3500):")
    print_pulse_summary(stepper, [T_CELL, B_CELL])

    fig, ax = plt.subplots(figsize=(5, 2))
    plot_cells_over_time(
        stepper=stepper,
        cell_types=[T_CELL, B_CELL],
        log_cells=False,
        legend_loc="upper right",
        ax=ax,
        ylim=(0, 6.2),
    )
    ax.get_legend().remove()  # type: ignore
    ax.set_title("Fig. 4e")
    plt.show()


def debug_fig_4f(refit: bool):
    """Fig. 4f: deterministic 3D trajectory in CD4 / CD8 / B space."""
    ana = get_3d_ana(refit)

    initial_cell_counts = [40, 4, 1]
    odeint_timepoints = np.linspace(0, 5000, 10000)

    sol = compute_trajectory(ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints)
    print("4f trajectory (cells per neighbourhood):")
    print_trajectory_summary(sol, ana.cell_types, odeint_timepoints)

    plot_3D_trajectory(ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints, logspace=False)
    plt.show()


def debug_fig_4g(refit: bool):
    """Fig. 4g: neighbour counts in the most / least proliferative states of each cell type."""
    np.random.seed(SEED)  # fixes the KDE jitter noise in plot_high_and_low_proliferation_states
    ana = get_3d_ana(refit)

    for i, cell_a in enumerate(ana.cell_types):
        counts, obs = ana.rnds.fetch(cell_type=cell_a)
        rates = ana.model.delta_cells(cell_counts=counts, mode="rates", return_order=ana.cell_types)[i]
        order = np.argsort(rates)

        low = counts.iloc[order[:K_STATES]]
        high = counts.iloc[order[-K_STATES:]]

        print(f"===== {cell_a}: {counts.shape[0]} cells, observed division rate {obs[DIVISION_COL].mean():.4f}")
        print(f"predicted rate (p+ - p-): min {rates.min():.4f}, max {rates.max():.4f}")
        print("mean neighbour counts in low / high proliferation states:")
        print(np.round(np.vstack([low.mean(), high.mean()]), 1), "(rows: low, high; cols:", list(counts.columns), ")\n")
        # expected (paper): high states -> many CD4, few B; low states -> few CD4, many B

    plot_high_and_low_proliferation_states(ana, ylim=0.12, xlim=120)
    plt.show()


PANELS = {
    "abc": debug_fig_4abc,
    "d": debug_fig_4d,
    "e": debug_fig_4e,
    "f": debug_fig_4f,
    "g": debug_fig_4g,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Fig. 4 panels. With no panel flags, runs all panels.")
    for name, func in PANELS.items():
        parser.add_argument(f"--{name}", action="store_true", help=func.__doc__.strip().splitlines()[0])
    parser.add_argument("--refit", action="store_true", help="refit the models instead of loading .cached_analyses/")
    return parser.parse_args()


def main():
    args = parse_args()

    selected = [name for name in PANELS if getattr(args, name)] or list(PANELS)
    print(f"panels: {selected}\n")

    for name in selected:
        print(f"########## Fig. 4{name} ##########")
        PANELS[name](refit=args.refit)


if __name__ == "__main__":
    main()
