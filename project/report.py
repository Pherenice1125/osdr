"""
Markdown report comparing the OSDR ODE with the ODE recovered by SINDy (written by part3_sindy.py --exp main).

The equations are written in LaTeX ($$ ... $$), which renders in the VS Code markdown preview and on GitHub.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from osdr_ode import CELL_TYPES, OSDRParams
from sindy_tools import PolyODE

TERMS_PER_LINE = 4
FIGURES = [
    ("Fig. 4f initial state: time series (solid OSDR, dashed SINDy)", "part3_main_timeseries.png"),
    ("2D projections of the Fig. 4f trajectory", "part3_main_projections.png"),
    ("3D trajectory: OSDR", "part3_main_3d_osdr.png"),
    ("3D trajectory: SINDy", "part3_main_3d_sindy.png"),
]


"""
LaTeX helpers:
"""


def tex_var(name: str) -> str:
    return rf"\mathrm{{{name}}}"


def tex_number(c: float, digits: int = 3) -> str:
    """3.1e-05 -> 3.10 \\times 10^{-5}; numbers between 0.01 and 100 are written plainly."""
    if c == 0:
        return "0"
    if 1e-2 <= abs(c) < 1e2:
        return f"{c:.{digits}g}"
    mantissa, exponent = f"{c:.{digits - 1}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def tex_monomial(exponents: np.ndarray, variables: list[str] = CELL_TYPES) -> str:
    parts = [tex_var(v) + (f"^{{{e}}}" if e > 1 else "") for v, e in zip(variables, exponents) if e > 0]
    return r"\,".join(parts)


def tex_sum(coefs, monomials) -> str:
    """Signed sum of `coef * monomial` terms, broken into lines of TERMS_PER_LINE terms (for an aligned block)."""
    terms = []
    for c, m in zip(coefs, monomials):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        body = tex_number(abs(c)) + (r"\," + m if m else "")
        terms.append((sign, body))
    if not terms:
        return "0"

    lines, line = [], []
    for k, (sign, body) in enumerate(terms):
        if k == 0:
            line.append(("-" if sign == "-" else "") + body)
        else:
            line.append(f"{sign} {body}")
        if len(line) == TERMS_PER_LINE:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return r" \\ &\quad ".join(lines)


def osdr_equations_tex(params: OSDRParams) -> list[str]:
    from itertools import combinations_with_replacement

    combos = [c for d in range(3) for c in combinations_with_replacement(range(3), d)]
    monomials = [tex_monomial(np.bincount(np.array(c, dtype=int), minlength=3)) if c else "" for c in combos]

    blocks = []
    for i, c in enumerate(params.cell_types):
        x = tex_var(c)
        rate = rf"\sigma(z_{{{x}}})"
        if params.truncate:
            rate = rf"\min\big({rate},\ {params.p_max[i]:.4f}\big)"
        ode = rf"\frac{{d{x}}}{{dt}} &= {x}\Big[{rate} - {params.p_minus[i]:.4f}\Big]"
        z = rf"z_{{{x}}} &= {tex_sum(params.beta[i], monomials)}"
        blocks.append("$$\n\\begin{aligned}\n" + ode + r" \\" + "\n" + z + "\n\\end{aligned}\n$$\n")
    return blocks


def poly_equations_tex(ode: PolyODE) -> list[str]:
    monomials = [tex_monomial(e) for e in ode.exponents]
    blocks = []
    for i, c in enumerate(CELL_TYPES):
        x = tex_var(c)
        body = rf"\frac{{d{x}}}{{dt}} &= {tex_sum(ode.coef[i], monomials)}"
        blocks.append("$$\n\\begin{aligned}\n" + body + "\n\\end{aligned}\n$$\n")
    return blocks


"""
Tables:
"""


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def md_term(name: str) -> str:
    return "`1`" if name == "1" else "`" + name.replace(" ", "·") + "`"


def coefficient_table(i: int, ode: PolyODE, ref: PolyODE, scaled_sindy, scaled_ref) -> str:
    """One equation: all 20 library terms, sorted by importance in the LS reference."""
    order = np.argsort(-np.abs(scaled_ref[i]))
    rows = []
    for k in order:
        a, b = scaled_sindy[i, k], scaled_ref[i, k]
        if a == 0:
            status = "**removed by SINDy**"
        elif np.sign(a) != np.sign(b):
            status = "kept, **sign differs**"
        else:
            status = "kept"
        rows.append([
            md_term(ode.feature_names[k]),
            int(ode.exponents[k].sum()),
            f"{b:+.4f}",
            f"{a:+.4f}" if a != 0 else "0",
            f"{ref.coef[i, k]:+.3e}",
            f"{ode.coef[i, k]:+.3e}" if a != 0 else "0",
            status,
        ])
    header = ["term", "degree", "OSDR (LS cubic), scaled", "SINDy, scaled", "OSDR (LS cubic), cell units",
              "SINDy, cell units", "status"]
    return md_table(header, rows)


def sign_symbol(v: float) -> str:
    return "+" if v > 0 else ("−" if v < 0 else "0")


"""
Report:
"""


def write_ode_comparison(
    path: Path,
    params: OSDRParams,
    ode: PolyODE,
    ref: PolyODE,
    res: dict,
    comparisons: dict,
    settings: dict,
) -> Path:
    coefs = comparisons["coefficients"]
    vf = comparisons["vector_field"]
    inter = comparisons["interactions"]
    origin = comparisons["origin"]
    beh = comparisons["behaviour"]

    n_osdr_params = params.beta.size + params.p_minus.size + (params.p_max.size if params.truncate else 0)
    out: list[str] = []
    w = out.append

    w("# OSDR ODE vs. SINDy ODE (Fig. 4f model)\n")
    w(f"Generated by `python project/part3_sindy.py --exp main` on {date.today().isoformat()}.\n")
    w(
        f"SINDy was trained on **{settings['n_traj']} time series** (from {settings['n_initial_states']} observed "
        f"neighbourhood compositions; trajectories that escaped were dropped), each of length "
        f"**{settings['t_len']:g}** sampled every **dt = {settings['dt']:g}**, generated by the OSDR ODE. "
        f"Library: all monomials up to degree 3 ({ode.coef.shape[1]} terms per equation); optimizer: STLSQ with "
        f"threshold **{settings['threshold']:g}** (chosen on held-out trajectories); states scaled by "
        f"1/{settings['state_scale']:g} during fitting.\n"
    )

    # ---------------------------------------------------------------- structure
    w("## 1. The two models at a glance\n")
    w(md_table(
        ["", "OSDR (ground truth)", "SINDy (recovered)"],
        [
            ["form", r"$X_i\,[\min(\sigma(\beta_i^\top\phi(X)),\,p_i^{\max}) - p_i^-]$",
             r"$\sum_k \xi_{ik}\,\theta_k(X)$, sparse cubic polynomial"],
            ["nonlinearity", "sigmoid of a quadratic polynomial, times $X_i$", "polynomial (degree ≤ 3)"],
            ["number of parameters", f"{n_osdr_params} (β: {params.beta.size}, p⁻: 3, p_max: 3)",
             f"{ode.n_terms} nonzero of {ode.coef.size}"],
            ["estimated from", "Ki67 + neighbour counts in one spatial snapshot (logistic regression)",
             "simulated time series of the OSDR ODE (sparse regression)"],
            ["$f(0)=0$ (no cells stay no cells)", "always ($X_i$ multiplies every term)",
             "only if constant terms are removed — here: "
             + ("yes" if np.allclose(origin["f0_est"], 0) else f"no, f(0) = {np.round(origin['f0_est'], 5)}")],
            ["cell numbers stay ≥ 0", "always ($X_i = 0$ is invariant)", "not guaranteed"],
            ["per-cell growth rate", f"bounded: between −p⁻ and p_max − p⁻", "unbounded outside the training data"],
        ],
    ))
    w("")

    # ---------------------------------------------------------------- equations
    w("## 2. The equations\n")
    w("### 2.1 OSDR (ground truth)\n")
    w(r"$\sigma(z) = 1/(1+e^{-z})$. Each $z$ is the logistic-regression score of the division probability.")
    w("")
    out.extend(osdr_equations_tex(params))
    w("")
    w(f"### 2.2 SINDy ({ode.n_terms} terms)\n")
    out.extend(poly_equations_tex(ode))
    w("")
    w("### 2.3 Why the formulas cannot be compared term by term\n")
    w(
        "OSDR is not a polynomial, so it has no 'true' cubic coefficients. To compare coefficients we use the "
        "**least-squares cubic approximation of the OSDR vector field** on the same training states (no sparsity): "
        "the best that a cubic polynomial can do, i.e. OSDR written in SINDy's language. Section 3 compares SINDy "
        "against it. Sections 4–6 compare SINDy against the OSDR ODE itself.\n"
    )

    # ---------------------------------------------------------------- coefficients
    w("## 3. Coefficients: SINDy vs. the best cubic approximation of OSDR\n")
    w(
        f"Relative coefficient error (scaled units): **{coefs['coef_err']:.3f}**. "
        f"Terms kept by SINDy: **{ode.n_terms} / {ode.coef.size}**.\n"
    )
    w(
        "*Scaled* coefficients are those of the model in units $X/"
        f"{settings['state_scale']:g}$, where every term is of order 1, so their sizes are comparable across terms "
        "(in cell units a cubic coefficient is tiny only because $X^3$ is huge). Rows are sorted by the size of "
        "the OSDR (LS cubic) coefficient.\n"
    )
    for i, c in enumerate(CELL_TYPES):
        w(f"<details><summary><b>d{c}/dt</b> — {int(np.count_nonzero(ode.coef[i]))} of {ode.coef.shape[1]} terms kept"
          "</summary>\n")
        w(coefficient_table(i, ode, ref, coefs["scaled_sindy"], coefs["scaled_ref"]))
        w("\n</details>\n")

    # ---------------------------------------------------------------- interactions
    w("## 4. Interactions: sign of the mean Jacobian $\\partial f_i / \\partial X_j$\n")
    w(
        "Averaged over the observed neighbourhood compositions. **+** means that more cells of type *j* "
        "increase the growth of type *i*, **−** that they decrease it. "
        f"Sign agreement: **{inter['agree']} / 9**.\n"
    )
    J, G = inter["J_true"], inter["J_est"]
    rows = []
    for i, c in enumerate(CELL_TYPES):
        cells = []
        for j in range(3):
            mark = "✓" if np.sign(J[i, j]) == np.sign(G[i, j]) else "✗"
            cells.append(f"{sign_symbol(J[i, j])} / {sign_symbol(G[i, j])} {mark}<br>"
                         f"<sub>{J[i, j]:+.4f} / {G[i, j]:+.4f}</sub>")
        rows.append([f"**{c}**"] + cells)
    w(md_table(["effect on ↓ of more →"] + [f"**{c}**" for c in CELL_TYPES], rows))
    w("\n(OSDR / SINDy)\n")

    # ---------------------------------------------------------------- vector field + origin
    w("## 5. Vector field and the origin\n")
    w(md_table(
        ["region", "relative error ‖f_SINDy − f_OSDR‖ / ‖f_OSDR‖", "median pointwise error", "mean cosine"],
        [[name, f"{e['rel_err']:.3f}", f"{e['median_pointwise_rel_err']:.3f}", f"{e['mean_cosine']:.3f}"]
         for name, e in vf.items()],
    ))
    w("")
    w(md_table(
        ["at the origin (no immune cells)", "OSDR", "SINDy"],
        [
            ["f(0)", str(np.round(origin["f0_true"], 5)), str(np.round(origin["f0_est"], 5))],
            ["eigenvalues of J(0)", str(np.round(origin["eig_true"], 5)), str(np.round(origin["eig_est"], 5))],
            ["origin stable?", "yes" if np.all(origin["eig_true"] < 0) else "no",
             "yes" if np.all(origin["eig_est"] < 0) else "no"],
        ],
    ))
    w("")

    # ---------------------------------------------------------------- behaviour
    w("## 6. Behaviour\n")
    rows = [["excitability threshold CD4(0) (CD8 = 4, B = 1)", f"{beh['threshold_true']:.1f}",
             f"{beh['threshold_est']:.1f}"]]
    for c, (p_true, t_true, p_est, t_est) in beh["peaks"].items():
        rows.append([f"{c} peak (time) from (40, 4, 1)", f"{p_true:.1f} (t = {t_true:.0f})",
                     f"{p_est:.1f} (t = {t_est:.0f})"])
    w(md_table(["", "OSDR", "SINDy"], rows))
    w("")
    w(
        f"Relative trajectory error from the Fig. 4f initial state (not used for training): "
        f"**{res['fig4f_err']:.3f}**. Median error over 20 held-out initial states: "
        f"**{res['test_median_err']:.3f}** ({res['test_escaped']:.0%} escaped).\n"
    )

    # ---------------------------------------------------------------- threshold + figures
    w("## 7. Threshold selection\n")
    w(md_table(
        ["threshold", "validation error", "terms", ""],
        [[f"{th:.0e}", f"{err:.4f}", n, "**chosen**" if th == settings["threshold"] else ""]
         for th, err, n in settings["sweep"]],
    ))
    w("")
    w("## 8. Figures\n")
    for caption, name in FIGURES:
        w(f"**{caption}**\n")
        w(f"![{caption}](../figures/{name})\n")

    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out))
    return path
