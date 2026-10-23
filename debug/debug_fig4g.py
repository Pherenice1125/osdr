"""
Step-by-step reproduction of Fig. 4g (high vs. low proliferation states of CD4 T, CD8 T and B cells).

Mirrors fig_4g() -> get_3d_tb_ana() -> plot_high_and_low_proliferation_states(), but bypasses
@persistent_cache so that breakpoints inside tdm are always hit.

Run with the "Python: Current File (step into tdm)" launch configuration.
"""

import numpy as np
import matplotlib.pyplot as plt

from tdm.analysis import Analysis
from tdm.cell_types import CD4_T_CELL, CD8_T_CELL, B_CELL
from tdm.paths import CACHED_ANALYSES_DIR
from tdm.plot.model.interpret import plot_high_and_low_proliferation_states
from tdm.preprocess.single_cell_df import CELL_TYPE_COL, DIVISION_COL
from tdm.raw.breast_mibi import read_single_cell_df, CELL_TYPE_DEFINITIONS_CD4_CD8_B

SEED = 0  # fixes the KDE jitter noise in plot_high_and_low_proliferation_states
K_STATES = 100  # hard-coded in plot_high_and_low_proliferation_states
CACHE_NAME = "debug_fig4g_tb3d.pkl"  # saved under .cached_analyses/
REFIT = False  # set True after changing tdm code to rebuild the analysis

CELL_TYPES = [CD4_T_CELL, CD8_T_CELL, B_CELL]


def load_data():
    """Step 1: Danenberg et al. single-cell table with CD4 / CD8 / B split."""
    df = read_single_cell_df(cell_type_definitions=CELL_TYPE_DEFINITIONS_CD4_CD8_B)

    print("n cells per type:\n", df[CELL_TYPE_COL].value_counts(), "\n")
    print("fraction dividing (Ki67 > threshold):\n", df.groupby(CELL_TYPE_COL)[DIVISION_COL].mean(), "\n")
    return df


def fit_analysis(df):
    """Step 2: same arguments as get_3d_tb_ana() in publications/first/analyses.py."""
    return Analysis(
        single_cell_df=df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        cell_types_to_model=CELL_TYPES,
        polynomial_dataset_kwargs={
            "degree": 2,
        },
        enforce_max_density=False,
        max_density_enforcer_power=8,
        verbose=True,
    )


def get_analysis():
    if not REFIT and (CACHED_ANALYSES_DIR / CACHE_NAME).exists():
        print(f"loading cached analysis: {CACHED_ANALYSES_DIR / CACHE_NAME}\n")
        return Analysis.load(CACHE_NAME)

    ana = fit_analysis(load_data())
    ana.dump(CACHE_NAME)
    return ana


def inspect_extreme_states(ana, cell_a):
    """Step 3: the numbers behind one column of Fig. 4g."""
    i = ana.cell_types.index(cell_a)

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


def main():
    np.random.seed(SEED)

    ana = get_analysis()
    print(ana, "\n")

    for cell_a in ana.cell_types:
        inspect_extreme_states(ana, cell_a)

    # Step 4: the actual figure (one figure per modelled cell type = one column of Fig. 4g)
    plot_high_and_low_proliferation_states(ana, ylim=0.12, xlim=120)
    plt.show()


if __name__ == "__main__":
    main()
