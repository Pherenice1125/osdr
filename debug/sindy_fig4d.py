"""
SINDy version of Fig. 4d (a single adaptive immune flare), side by side with the OSDR original.

OSDR fits P(division | neighbourhood) with a logistic regression on polynomial features of (T, B).
Here the same features / targets are fitted with SINDy's sequentially-thresholded least squares (STLSQ):
a sparse *linear* model of the division probability. Everything else is kept identical
(death = mean division rate, truncation at the max observed rate, maximal density enforcer,
the stochastic TissueStep simulator, seed and initial state), so the difference between the panels
is only the regression step.

    python debug/sindy_fig4d.py
    python debug/sindy_fig4d.py --threshold 0.75 --refit

Reuses the cached 2D T-B analysis from debug_fig4.py (.cached_analyses/debug_fig4_tb2d.pkl).
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from tdm.cell_types import T_CELL, B_CELL
from tdm.dataset.dataset import Dataset
from tdm.model.model import Model
from tdm.plot.tissue.stepper import plot_cells_over_time
from tdm.simulate.generate_distribution import simulate_one_tissue
from tdm.utils import inv_log2_1p

from debug_fig4 import SEED, TISSUE_SIZE, get_2d_ana, print_pulse_summary, print_trajectory_summary

STATE0 = (4.5, 1.0)  # (T, B) in log2(1 + n) units per neighbourhood, as in f4.py
N_STEPS = 2000
OUT_PATH = Path(__file__).parent / "sindy_fig4d.png"


"""
SINDy model:
"""


def stlsq(X: np.ndarray, y: np.ndarray, threshold: float, n_iter: int = 10) -> np.ndarray:
    """Sequentially thresholded least squares (Brunton et al. 2016), on unit-norm columns.

    Columns are normalised because the raw library spans ~5 orders of magnitude (1 ... B^2),
    so the threshold is applied to coefficients of comparable scale.
    """
    norms = np.linalg.norm(X, axis=0)
    Xn = X / norms

    coef = np.linalg.lstsq(Xn, y, rcond=None)[0]
    for _ in range(n_iter):
        keep = np.abs(coef) >= threshold
        coef[~keep] = 0
        if not keep.any():
            break
        coef[keep] = np.linalg.lstsq(Xn[:, keep], y, rcond=None)[0]

    return coef / norms


class STLSQFit:
    """Minimal fitted-model object: holds coefficients and predicts on a feature dataframe."""

    def __init__(self, feature_names: list[str], coef: np.ndarray) -> None:
        self.feature_names = feature_names
        self.coef = coef

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == self.feature_names
        # a linear model is not bounded like the logistic one: clip to valid probabilities
        return np.clip(features.values @ self.coef, 0, 1)

    def equation(self) -> str:
        return " ".join(f"{c:+.3g}*{n}" for c, n in zip(self.coef, self.feature_names) if c != 0) or "0"


class SINDyModel(Model):
    """A :class:`~tdm.model.Model` whose division probability is a sparse polynomial fitted with STLSQ."""

    def __init__(self, dataset: Dataset, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = threshold
        super().__init__(dataset, **kwargs)

    def fit(self, features: pd.DataFrame, obs: pd.Series, cell_type: str) -> STLSQFit:
        coef = stlsq(features.values, obs.values.astype(float), self.threshold)
        return STLSQFit(list(features.columns), coef)

    def _predict(
        self,
        cell_type: str,
        obs: Literal["death", "division"],
        features: pd.DataFrame,
    ) -> np.ndarray:
        if obs in ["death", "division"]:
            return np.array(self.models[cell_type][obs].predict(features))
        else:
            raise ValueError(f"Invalid argument: _predict(obs={obs})")

    def parameters(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        return self.models[cell_type][obs].coef


"""
Comparison:
"""


def print_equations(ana, sindy_model: SINDyModel):
    for cell_type in ana.cell_types:
        logit = ana.model.models[cell_type]["division"]
        logit_terms = " ".join(f"{c:+.3g}*{n}" for c, n in zip(logit.params.values, logit.model.exog_names))
        p_death = sindy_model.models[cell_type]["death"].p

        print(f"===== {cell_type}  (death rate = mean division rate = {p_death:.4f})")
        print(f"OSDR : p_div = sigmoid({logit_terms})")
        print(f"SINDy: p_div = {sindy_model.models[cell_type]['division'].equation()}")
        print(f"d{cell_type}/dt = {cell_type} * (p_div - {p_death:.4f}) + max-density correction\n")


def deterministic_trajectory(model: Model, cell_types: list[str], timepoints: np.ndarray) -> np.ndarray:
    f = model.get_delta_cells_func(predicted_cell_types=cell_types)
    return odeint(lambda x, t: f(x), inv_log2_1p(np.array(STATE0)), timepoints)


def simulate(ana, model: Model):
    np.random.seed(SEED)
    return simulate_one_tissue(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        model=model,
        neighborhood_size=ana.neighborhood_size,
        n_steps=N_STEPS,
        tissue_width=TISSUE_SIZE,
        tissue_height=TISSUE_SIZE,
        state0=STATE0,
        return_stepper=True,
        verbose=True,
        diffusion_coef=0,
    )


def main(threshold: float, refit: bool):
    ana = get_2d_ana(refit)

    # same dataset, death estimate, truncation and max-density enforcer as the OSDR model:
    sindy_model = ana._fit_model(
        model_class=SINDyModel,
        model_kwargs={**ana._model_kwargs, "threshold": threshold},
    )
    print_equations(ana, sindy_model)

    models = {"OSDR (logistic)": ana.model, f"SINDy (STLSQ, threshold={threshold})": sindy_model}
    cell_types = [T_CELL, B_CELL]
    timepoints = np.arange(N_STEPS + 1, dtype=float)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 4.5), sharex=True)
    for col, (name, model) in enumerate(models.items()):
        print(f"##### {name}")

        sol = deterministic_trajectory(model, cell_types, timepoints)
        print("deterministic ODE (cells per neighbourhood):")
        print_trajectory_summary(sol, cell_types, timepoints)

        stepper = simulate(ana, model)
        print("stochastic simulation:")
        print_pulse_summary(stepper, cell_types)

        ax = axes[0, col]
        plot_cells_over_time(
            stepper=stepper,
            cell_types=cell_types,
            log_cells=False,
            legend_loc="upper right",
            ax=ax,
            ylim=(0, 6.2),
            n_steps=1500,
        )
        ax.set_title(f"Fig. 4d - {name}\nstochastic simulation", fontsize=9)

        ax = axes[1, col]
        for i, cell_type in enumerate(cell_types):
            ax.plot(timepoints[:1501], sol[:1501, i], label=cell_type)
        ax.set_title("deterministic ODE", fontsize=9)
        ax.set_xlabel("time step")
        ax.set_ylabel("cells / neighbourhood")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.5, help="STLSQ threshold on unit-norm coefficients")
    parser.add_argument("--refit", action="store_true", help="refit the cached OSDR analysis")
    args = parser.parse_args()
    main(args.threshold, args.refit)
