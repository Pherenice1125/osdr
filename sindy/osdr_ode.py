"""
Shared utilities: the 3D OSDR ODE (CD4 T, CD8 T, B cells) behind Fig. 4f, as a fast numpy function.

The fitted model (logistic regression on quadratic neighbour-count features) defines, for each cell type i:

    dX_i/dt = X_i * [ min( sigmoid(beta_i . phi(X)), p_max_i ) - p_minus_i ]

    phi(X)     = [1, CD4, CD8, B, CD4^2, CD4 CD8, CD4 B, CD8^2, CD8 B, B^2]
    beta_i     = logistic-regression coefficients (division probability)
    p_minus_i  = constant death rate (mean division rate of type i)
    p_max_i    = maximal predicted division rate in the data (truncate_division_rate=True)

The parameters are read once from the cached tdm Analysis (debug/debug_fig4.py --f) and stored in a small
json file, so later scripts do not need to unpickle the ~570MB analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace, field
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.special import expit

SINDY_DIR = Path(__file__).resolve().parent
CACHE_DIR = SINDY_DIR / "_cache"
PARAMS_PATH = CACHE_DIR / "osdr_3d_params.json"
STATES_PATH = CACHE_DIR / "osdr_3d_data_states.npy"

ANALYSIS_CACHE_NAME = "debug_fig4_tb3d.pkl"  # created by debug/debug_fig4.py (under .cached_analyses/)

CELL_TYPES = ["CD4", "CD8", "B"]
CELL_COLORS = {"CD4": "#9D65AB", "CD8": "#3B76BC", "B": "#3E754E"}

# Fig. 4f settings:
FIG4F_X0 = np.array([40.0, 4.0, 1.0])
FIG4F_T_END = 5000.0


@dataclass(frozen=True)
class OSDRParams:
    cell_types: list[str]
    feature_names: list[str]  # names of phi(X), e.g. "CD4 B"
    beta: np.ndarray  # (n_cell_types, n_features)
    p_minus: np.ndarray  # (n_cell_types,)
    p_max: np.ndarray  # (n_cell_types,)
    truncate: bool = True
    label: str = "OSDR"
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "cell_types": self.cell_types,
            "feature_names": self.feature_names,
            "beta": self.beta.tolist(),
            "p_minus": self.p_minus.tolist(),
            "p_max": self.p_max.tolist(),
            "truncate": self.truncate,
        }

    @staticmethod
    def from_json(d: dict) -> "OSDRParams":
        return OSDRParams(
            cell_types=d["cell_types"],
            feature_names=d["feature_names"],
            beta=np.array(d["beta"]),
            p_minus=np.array(d["p_minus"]),
            p_max=np.array(d["p_max"]),
            truncate=d["truncate"],
        )


"""
Loading parameters from the fitted tdm model:
"""


def _export_from_analysis(n_states: int = 5000, seed: int = 0):
    """Read beta, p-, p_max and a sample of observed neighbourhood states from the cached Analysis."""
    from tdm.analysis import Analysis  # heavy import, only needed once

    print(f"exporting OSDR parameters from .cached_analyses/{ANALYSIS_CACHE_NAME} (one-time, ~1 min)...")
    ana = Analysis.load(ANALYSIS_CACHE_NAME)
    model = ana.model
    assert ana.cell_types == CELL_TYPES, ana.cell_types
    assert model.maximal_density_enforcer is None  # get_3d_tb_ana uses enforce_max_density=False

    feature_names = list(model.ds.feature_names)
    params = OSDRParams(
        cell_types=list(ana.cell_types),
        feature_names=feature_names,
        beta=np.array([model.models[c]["division"].params[feature_names].values for c in ana.cell_types]),
        p_minus=np.array([model.models[c]["death"].p for c in ana.cell_types]),
        p_max=np.array([model._max_observed_fluxes[c] for c in ana.cell_types]),
        truncate=bool(model.truncate_division_rate),
    )

    # observed neighbourhood compositions (CD4, CD8, B counts) of all modelled cells:
    states = np.vstack([ana.rnds.fetch(c)[0][ana.cell_types].values for c in ana.cell_types])
    rng = np.random.default_rng(seed)
    states = states[rng.choice(len(states), size=min(n_states, len(states)), replace=False)]

    CACHE_DIR.mkdir(exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params.to_json(), indent=2))
    np.save(STATES_PATH, states)
    print(f"saved {PARAMS_PATH.name} and {STATES_PATH.name}\n")


def load_params() -> OSDRParams:
    if not PARAMS_PATH.exists():
        _export_from_analysis()
    return OSDRParams.from_json(json.loads(PARAMS_PATH.read_text()))


def load_data_states() -> np.ndarray:
    """A sample of observed (CD4, CD8, B) neighbourhood compositions from the Danenberg data."""
    if not STATES_PATH.exists():
        _export_from_analysis()
    return np.load(STATES_PATH)


"""
Parameter changes (part 2):
"""


def with_death_scale(params: OSDRParams, scale: float) -> OSDRParams:
    """Multiply the constant death rates p- of all cell types by `scale`."""
    return replace(params, p_minus=params.p_minus * scale, label=f"p- x {scale:g}")


def with_b_inhibition_scale(params: OSDRParams, scale: float) -> OSDRParams:
    """Multiply every B-containing term (B, CD4 B, CD8 B, B^2) in the T cell (CD4, CD8) equations by `scale`.

    scale = 0 removes any effect of B cells on T cell division; scale = 1 is the fitted model.
    """
    beta = params.beta.copy()
    b_cols =[j for j, name in enumerate(params.feature_names) if "B" in name.replace("^2", "").split()]
    t_rows = [params.cell_types.index(c) for c in ["CD4", "CD8"]]
    beta[np.ix_(t_rows, b_cols)] *= scale
    return replace(params, beta=beta, label=f"B->T x {scale:g}")


def with_truncation(params: OSDRParams, truncate: bool) -> OSDRParams:
    return replace(params, truncate=truncate, label=f"truncate={truncate}")


"""
The ODE:
"""


def poly2_features(X: np.ndarray) -> np.ndarray:
    """phi(X) for X with shape (n, 3), same ordering as tdm's MyPolynomialFeatures(degree=2)."""
    X = np.atleast_2d(X)
    combs = [c for d in range(3) for c in combinations_with_replacement(range(X.shape[1]), d)]
    return np.column_stack([np.prod(X[:, list(c)], axis=1) for c in combs])


def sigmoid(z):
    return expit(z)


def division_rates(params: OSDRParams, X: np.ndarray) -> np.ndarray:
    """p+(X) for each cell type, shape (n, n_cell_types)."""
    p = sigmoid(poly2_features(X) @ params.beta.T)
    if params.truncate:
        p = np.minimum(p, params.p_max)
    return p


def rhs(params: OSDRParams, X: np.ndarray) -> np.ndarray:
    """dX/dt evaluated at states X, shape (n, 3) -> (n, 3)."""
    X = np.atleast_2d(X)
    return X * (division_rates(params, X) - params.p_minus)


ESCAPE_CELLS = 1000.0  # far outside the data (max ~260 cells per neighbourhood)


def integrate(f, x0, t_eval: np.ndarray, escape: float = ESCAPE_CELLS) -> np.ndarray:
    """Integrate dx/dt = f(x) (f maps a state vector to a vector).

    Returns shape (len(t_eval), d). If any variable leaves [-escape, escape] the integration stops and the
    remaining time points are NaN (the trajectory escaped the region where the model is meaningful).
    """

    def escaped(t, x):
        return escape - np.abs(x).max()

    escaped.terminal = True

    sol = solve_ivp(
        lambda t, x: f(x),
        (t_eval[0], t_eval[-1]),
        np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-8,
        events=escaped,
    )
    out = np.full((len(t_eval), len(x0)), np.nan)
    out[: sol.y.shape[1]] = sol.y.T
    return out


def escape_time(sol: np.ndarray, t_eval: np.ndarray) -> float | None:
    """Time at which an integrated trajectory escaped (None if it did not)."""
    bad = np.isnan(sol).any(axis=1)
    return float(t_eval[np.argmax(bad)]) if bad.any() else None


def simulate(params: OSDRParams, x0, t_eval: np.ndarray, escape: float = ESCAPE_CELLS) -> np.ndarray:
    """Integrate the OSDR ODE; returns an array with shape (len(t_eval), 3), NaN after an escape."""
    return integrate(lambda x: rhs(params, x)[0], x0, t_eval, escape)


"""
Printing the equations:
"""


def sympy_equations(params: OSDRParams, digits: int = 4) -> list[tuple[sp.Eq, sp.Eq]]:
    """For each cell type: (dX/dt = X [min(sigma(z), p_max) - p-],  z = beta . phi(X))."""
    syms = sp.symbols(params.cell_types)
    phi = [sp.Integer(1)] + list(syms)
    phi += [syms[i] * syms[j] for i in range(len(syms)) for j in range(i, len(syms))]
    t = sp.Symbol("t")
    sigma = sp.Function("sigma")

    eqs = []
    for i, x in enumerate(syms):
        z_sym = sp.Symbol(f"z_{x}")
        z = sum(sp.Float(b, digits) * f for b, f in zip(params.beta[i], phi))
        p_div = sigma(z_sym)
        if params.truncate:
            p_div = sp.Min(p_div, sp.Float(params.p_max[i], digits))
        ode = sp.Eq(sp.Derivative(sp.Function(str(x))(t), t), x * (p_div - sp.Float(params.p_minus[i], digits)))
        eqs.append((ode, sp.Eq(z_sym, z)))
    return eqs


def format_equations(params: OSDRParams) -> str:
    lines = []
    for i, c in enumerate(params.cell_types):
        terms = " ".join(
            f"{b:+.4e}" + ("" if name == "1" else f"*{name.replace(' ', '*')}")
            for b, name in zip(params.beta[i], params.feature_names)
        )
        trunc = f"min( sigmoid(z_{c}), {params.p_max[i]:.4f} )" if params.truncate else f"sigmoid(z_{c})"
        lines.append(f"d{c}/dt = {c} * [ {trunc} - {params.p_minus[i]:.4f} ]")
        lines.append(f"    z_{c} = {terms}")
    return "\n".join(lines)
