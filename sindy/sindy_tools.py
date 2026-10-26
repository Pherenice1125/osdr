"""
SINDy fitting and the comparisons between the OSDR ODE and a SINDy ODE (part 3).

The OSDR ODE is X_i [sigmoid(poly2) - p-], which is not a polynomial, so the SINDy model (a sparse cubic
polynomial) cannot reproduce it exactly. We therefore compare functions and behaviour, not formulas:

    1. coefficients     vs. the best (least-squares) cubic approximation of the OSDR vector field
    2. vector field      relative error and direction (cosine) on sets of states
    3. interactions      sign of the mean Jacobian d f_i / d X_j  (activation / inhibition network)
    4. local dynamics    eigenvalues of the Jacobian at the origin (the stable "no immune cells" state)
    5. behaviour         trajectories, pulse peaks and the excitability threshold
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pysindy as ps

from osdr_ode import CELL_TYPES, OSDRParams, integrate, rhs, escape_time


"""
A fast evaluator for polynomial ODEs (SINDy models and least-squares references):
"""


@dataclass
class PolyODE:
    """dX/dt = coef @ theta(X), theta = monomials given by `exponents` (n_terms, d)."""

    feature_names: list[str]  # e.g. ["1", "CD4", "CD4 CD8", "B^2", ...]
    exponents: np.ndarray
    coef: np.ndarray  # (d, n_terms)
    label: str = "SINDy"

    @staticmethod
    def exponents_from_names(names: list[str], variables: list[str]) -> np.ndarray:
        E = np.zeros((len(names), len(variables)), dtype=int)
        for k, name in enumerate(names):
            if name == "1":
                continue
            for token in name.split():
                m = re.fullmatch(r"(\w+?)(?:\^(\d+))?", token)
                var, power = m.group(1), int(m.group(2) or 1)
                E[k, variables.index(var)] += power
        return E

    @staticmethod
    def from_sindy(model: ps.SINDy, variables: list[str] = CELL_TYPES, label: str = "SINDy") -> "PolyODE":
        names = model.get_feature_names()
        return PolyODE(names, PolyODE.exponents_from_names(names, variables), np.array(model.coefficients()), label)

    def theta(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.prod(X[:, None, :] ** self.exponents[None, :, :], axis=2)

    def rhs(self, X: np.ndarray) -> np.ndarray:
        return self.theta(X) @ self.coef.T

    def simulate(self, x0, t_eval: np.ndarray) -> np.ndarray:
        return integrate(lambda x: self.rhs(x)[0], x0, t_eval)

    @property
    def n_terms(self) -> int:
        return int(np.count_nonzero(self.coef))

    def format(self, variables: list[str] = CELL_TYPES, precision: int = 4) -> str:
        lines = []
        for i, v in enumerate(variables):
            terms = [
                f"{c:+.{precision}e}" + ("" if n == "1" else f" {n}")
                for c, n in zip(self.coef[i], self.feature_names)
                if c != 0
            ]
            lines.append(f"d{v}/dt = " + (" ".join(terms) if terms else "0"))
        return "\n".join(lines)


def cubic_library() -> ps.PolynomialLibrary:
    return ps.PolynomialLibrary(degree=3, include_bias=True)


def least_squares_reference(params: OSDRParams, states: np.ndarray, label="LS cubic fit of OSDR") -> PolyODE:
    """Best cubic polynomial approximation of the OSDR vector field on `states` (no sparsity).

    This is the "ground truth expressed in SINDy's language" used to compare coefficients.
    """
    lib = cubic_library().fit(states)
    names = lib.get_feature_names(CELL_TYPES)
    ref = PolyODE(names, PolyODE.exponents_from_names(names, CELL_TYPES), np.zeros((3, len(names))), label)
    Theta = ref.theta(states)
    F = rhs(params, states)
    # column scaling for a well-conditioned least-squares problem (cubic terms are ~1e5):
    scale = np.linalg.norm(Theta, axis=0)
    ref.coef = np.linalg.lstsq(Theta / scale, F, rcond=None)[0].T / scale
    return ref


"""
Training data and fitting:
"""


def make_training_data(f, x0s: np.ndarray, t_len: float, dt: float = 1.0):
    """Integrate dx/dt = f(x) from each initial state for t_len; drops trajectories that escape."""
    t = np.arange(0, t_len + dt / 2, dt)
    trajs = []
    for x0 in x0s:
        sol = integrate(f, x0, t)
        if escape_time(sol, t) is None:
            trajs.append(sol)
    return trajs, t


STATE_SCALE = 50.0  # cells per neighbourhood; fitting is done on X / STATE_SCALE


def fit_sindy(trajs: list[np.ndarray], dt: float, threshold: float, alpha: float = 1e-6) -> PolyODE:
    """Sparse cubic SINDy fit; derivatives are estimated from the time series by finite differences.

    The states are divided by STATE_SCALE before fitting so that all library terms are O(1) (otherwise
    CD4^3 ~ 1e5 and no threshold is meaningful), and the coefficients are converted back to cell units:
    a term with total degree k gets coef * s / s^k.
    """
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=50),
        feature_library=cubic_library(),
        differentiation_method=ps.FiniteDifference(order=2),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit([x / STATE_SCALE for x in trajs], t=dt, feature_names=CELL_TYPES)

    ode = PolyODE.from_sindy(model)
    degree = ode.exponents.sum(axis=1)
    ode.coef = ode.coef * STATE_SCALE / STATE_SCALE ** degree[None, :]
    ode.label = f"SINDy (threshold {threshold:g})"
    return ode


def fit_sindy_select_threshold(
    trajs: list[np.ndarray],
    dt: float,
    val_states: np.ndarray,
    val_f: np.ndarray,
    thresholds=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    tolerance: float = 1.1,
) -> tuple[PolyODE, float, list[tuple[float, float, int]]]:
    """Fit for several thresholds; keep the sparsest model whose validation error is within `tolerance` x best.

    The validation error is the relative error of the vector field on held-out states (from other trajectories).
    """
    results = []
    for th in thresholds:
        ode = fit_sindy(trajs, dt, th)
        err = np.linalg.norm(ode.rhs(val_states) - val_f) / np.linalg.norm(val_f)
        results.append((th, err, ode.n_terms, ode))

    best_err = min(r[1] for r in results)
    candidates = [r for r in results if r[1] <= tolerance * best_err and r[2] > 0]
    th, err, n, ode = min(candidates, key=lambda r: r[2])
    ode.label = "SINDy"
    return ode, th, [(r[0], r[1], r[2]) for r in results]


"""
Comparisons:
"""


def vector_field_errors(f_true, f_est, states: np.ndarray) -> dict:
    F, G = f_true(states), f_est(states)
    norms = np.linalg.norm(F, axis=1)
    keep = norms > 1e-3 * norms.max()  # cosine is undefined where the field vanishes
    cos = np.sum(F[keep] * G[keep], axis=1) / (norms[keep] * np.linalg.norm(G[keep], axis=1) + 1e-300)
    return {
        "rel_err": float(np.linalg.norm(G - F) / np.linalg.norm(F)),
        "median_pointwise_rel_err": float(np.median(np.linalg.norm(G - F, axis=1)[keep] / norms[keep])),
        "mean_cosine": float(np.mean(cos)),
    }


def mean_jacobian(f, states: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Mean over states of J_ij = d f_i / d X_j (central differences)."""
    d = states.shape[1]
    J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        J[:, j] = ((f(states + e) - f(states - e)) / (2 * h)).mean(axis=0)
    return J


def jacobian_at(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    return mean_jacobian(f, np.atleast_2d(np.asarray(x, dtype=float)), h)


def trajectory_error(sol_true: np.ndarray, sol_est: np.ndarray) -> float:
    """Relative L2 error over the whole time series (inf if the estimate escaped)."""
    if np.isnan(sol_est).any():
        return np.inf
    return float(np.linalg.norm(sol_est - sol_true) / np.linalg.norm(sol_true))


def excitability_threshold(simulate_from, cd8: float = 4.0, b: float = 1.0, grid=np.linspace(1, 40, 79)) -> float:
    """Smallest CD4(0) for which CD4 grows (instead of decaying) - see part 2."""
    t = np.linspace(0, 300, 301)
    for v in grid:
        sol = simulate_from(np.array([v, cd8, b]), t)
        if np.isnan(sol).any() or sol[:, 0].max() - v > 1e-6:
            return float(v)
    return np.nan
