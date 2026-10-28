# SINDy on the OSDR Fig. 4f model

This folder studies the 3-variable ODE behind **Fig. 4f** of Somer, Mannor & Alon, *Temporal tissue dynamics from a spatial snapshot* (Nature, 2026). The three variables are the numbers of CD4 T, CD8 T and B cells per 80 µm neighbourhood. The folder does three things:

1. **Part 1**: writes the fitted OSDR ODE out explicitly and simulates its time series.
2. **Part 2**: changes model parameters and shows how the immune pulse responds.
3. **Part 3**: generates time series from the OSDR ODE, recovers an ODE from them with SINDy, and compares the two.

> **Scope.** The SINDy training data is produced by the OSDR ODE itself. Part 3 therefore tests how well SINDy recovers a known ODE. It does **not** independently validate the biological conclusions of Fig. 4.

## The model

For each cell type $i \in \{\text{CD4}, \text{CD8}, \text{B}\}$:

$$
\frac{dX_i}{dt} = X_i\Big[\min\big(\sigma(\boldsymbol\beta_i^\top\boldsymbol\phi(X)),\ p_i^{\max}\big) - p_i^-\Big]
$$

| Symbol | Meaning |
| --- | --- |
| $\boldsymbol\phi(X)$ | quadratic features `1, CD4, CD8, B, CD4², CD4·CD8, CD4·B, CD8², CD8·B, B²` |
| $\boldsymbol\beta_i$ | logistic-regression coefficients for the division probability |
| $p_i^-$ | constant death rate (the mean division rate of type $i$) |
| $p_i^{\max}$ | largest division rate predicted on the data |

Time is measured in OSDR time units: one unit is roughly the time a cell stays Ki67-positive, about a few hours.

## Setup

Run all commands from the repository root inside the `osdr` conda environment.

```bash
conda activate osdr
pip install -e .                    # installs tdm and its dependencies, including pysindy>=2.0
python debug/debug_fig4.py --g      # one-time: downloads the data and fits/caches the 3D model
```

The second command opens Fig. 4g windows when it finishes; close them to end the script. It does two things:

- It downloads the Danenberg et al. IMC dataset to `data/breast_mibi/` (about 850 MB).
- It fits the CD4–CD8–B model and caches it as `.cached_analyses/debug_fig4_tb3d.pkl`.

The first time any script in this folder runs, it reads the fitted parameters from that cache. It then writes them to `project/_cache/`: a small JSON file with the parameters, plus a sample of observed neighbourhood states. Every later run uses only these small files.

## Quick start

```bash
python project/part1_timeseries.py --verify   # print the ODEs; check them against the original tdm model
python project/part2_params.py                # parameter experiments
python project/part3_sindy.py                 # all SINDy experiments
```

Every script saves its figures to `project/figures/`. Add `--show` to also open them in a window. Use `--help` to list a script's options.

## Files

| File | Contents |
| --- | --- |
| `osdr_ode.py` | The OSDR ODE in numpy (`rhs`, `simulate`), parameter loading, parameter modifiers (`with_death_scale`, `with_b_inhibition_scale`, `with_truncation`), equation printing |
| `sindy_tools.py` | SINDy fitting with automatic threshold selection, a fast polynomial-ODE class (`PolyODE`), and the comparison metrics |
| `plotting.py` | Shared plotting helpers (time series, 3D trajectory) |
| `part1_timeseries.py` | Part 1 |
| `part2_params.py` | Part 2 |
| `part3_sindy.py` | Part 3 |

## Part 1: the OSDR ODE

```bash
python project/part1_timeseries.py [--verify] [--show]
```

- Prints the three ODEs with their fitted coefficients, in three forms: plain text, sympy, and LaTeX ready to paste into a report.
- `--verify` compares `osdr_ode.rhs` with the original tdm model on 4000 states, and compares the simulated trajectory with `tdm.simulate.deterministic.compute_trajectory`.
- Simulates from the Fig. 4f initial state `(CD4, CD8, B) = (40, 4, 1)`.

**Outputs:**

- `part1_timeseries.png`: number of cells over time for all three cell types.
- `part1_trajectory_3d.png`: the Fig. 4f 3D trajectory, with its projections drawn in gray.

## Part 2: parameter changes

```bash
python project/part2_params.py [--exp x0|death|binhib|truncate] [--show]
```

`--exp` can be repeated. Without it, all four experiments run.

| `--exp` | What changes | What it shows |
| --- | --- | --- |
| `x0` | initial CD4 count, 2 to 40 | excitability threshold: no pulse below about 9 CD4 cells |
| `death` | all death rates $p^-$ multiplied by 0.5 to 1.5 | pulse size and duration; trajectories escape to unbounded growth for scales ≤ 0.75 |
| `binhib` | every B-containing term in the CD4 and CD8 equations multiplied by 0 to 2 | how much the T-cell dynamics depend on B cells |
| `truncate` | division-rate cap $p^{\max}$ on or off | no effect on the Fig. 4f trajectory, because rates along it stay below the cap |

Each experiment prints a table of peak heights, peak times and final values. It also saves `part2_<exp>.png`: the top row shows time series, the bottom row shows summary curves.

**Escaped trajectories.** If any variable exceeds 1000 cells, integration stops and the remaining time points are set to NaN. The tables report this as "escaped". The observed data never goes above about 260 cells per neighbourhood, so values like these mean the model is extrapolating.

## Part 3: SINDy

```bash
python project/part3_sindy.py [--exp main|length|ntraj|params|control] [--show]
```

### How SINDy is set up

- **Training data:** OSDR trajectories started from observed neighbourhood compositions in the Danenberg data, sampled with `dt = 1`. Defaults: 100 trajectories of length 2000.
- **Held-out sets:** 20 trajectories for choosing the threshold, and 20 initial states for testing. The Fig. 4f initial state is never used for training, except in the `ntraj` row "Fig. 4f only", so the Fig. 4f comparison is out-of-sample.
- **Library and optimizer:** cubic polynomial library, STLSQ optimizer, derivatives estimated from the time series by second-order finite differences.
- **Scaling:** states are divided by 50 before fitting, then the coefficients are converted back to cell units. Without this, terms like CD4³ (around 1e5) make the sparsity threshold meaningless.
- **Threshold selection:** the script fits several thresholds and keeps the sparsest model whose validation vector-field error is within 10% of the best.

### Experiments

| `--exp` | Question | Main output |
| --- | --- | --- |
| `main` | What ODE does SINDy find, and how close is it to OSDR? | printed SINDy ODE and comparisons; `part3_main_timeseries.png`, `part3_main_3d_osdr.png`, `part3_main_3d_sindy.png`, `part3_main_projections.png`; `results/main_coefficients.csv` |
| `length` | How long must each time series be? (25 to 3200) | `part3_length.png` and a metrics table |
| `ntraj` | How many trajectories are needed? (1 to 300, and "Fig. 4f only") | `part3_ntraj.png` and a metrics table |
| `params` | Does SINDy still work after the Part 2 parameter changes? | `part3_params.png`; excitability thresholds for OSDR and SINDy |
| `control` | Is the error caused by SINDy itself or by the sigmoid not being a polynomial? | `part3_control.png` and a coefficient-error comparison |

### Comparing a SINDy ODE with the OSDR ODE

The two ODEs have different forms: OSDR uses a sigmoid, SINDy a sparse cubic polynomial. `main` therefore compares functions and behaviour rather than formulas:

1. **Coefficients.** SINDy's coefficients are compared with the *least-squares cubic approximation* of the OSDR vector field on the same training states, i.e. OSDR expressed in SINDy's library. Errors are computed in scaled units so that all terms are comparable.
2. **Vector field.** Relative error and mean cosine similarity of $f$, reported on three sets: training states, observed data states, and states beyond the data.
3. **Interactions.** The sign of the mean Jacobian $\partial f_i/\partial X_j$, i.e. which cell type activates or inhibits which.
4. **Origin.** $f(0)$ and the eigenvalues of the Jacobian at the no-immune-cell state.
5. **Behaviour.** Excitability threshold, pulse peaks, and trajectory errors from held-out initial states.

### Metrics in the tables

| Column | Meaning |
| --- | --- |
| `n_terms` | number of nonzero SINDy coefficients (out of 60) |
| `threshold` | chosen STLSQ threshold |
| `vf_train`, `vf_data` | relative vector-field error on training states and on observed data states |
| `cos_data` | mean cosine between the OSDR and SINDy vector fields on data states |
| `fig4f_err` | relative L2 error of the trajectory from (40, 4, 1); `inf` if SINDy escaped |
| `test_median_err` | median trajectory error over the 20 held-out initial states |
| `test_escaped` | fraction of held-out trajectories that escaped |

## Notes and caveats

- **Validation scope.** This folder tests SINDy against a known ODE. It does not validate the OSDR biology: SINDy only ever sees data generated by the OSDR ODE.
- **Threshold choice.** Results depend on the threshold rule. Some unexpected points come from it rather than from the data, for example `t_len = 50` doing worse than `t_len = 25`, and 300 trajectories doing worse than 100. When this happens, compare with the threshold sweep printed by `main`.
- **Extrapolation.** Neither model is reliable outside the observed data. The OSDR 3D model has no maximal-density correction, and SINDy does not enforce $f(0) = 0$ or positivity.
- **Regenerating cached parameters.** If the cached 3D analysis is refit (`python debug/debug_fig4.py --g --refit`), delete `project/_cache/` so the parameters are exported again.
- **Debugging.** Open any script and use the launch configuration "Python: Current File (step into tdm)" in `.vscode/launch.json`.
