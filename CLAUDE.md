# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A STAT 548 qualifying paper, "Extended Ensemble Kalman Methods", plus the simulation code behind it. The paper critiques and reimplements the GEnKF and PEnKF (ensemble Kalman filters that also handle unknown static parameters) on the skewed, zero-inflated rainfall model of Katzfuss et al. 2020. There is no test suite, linter, or build system for the code.

## Paper (LaTeX)

- Main file is `qp1.tex`. It uses `shortex.sty` (third-party macro package) and `qp.sty` (personal additions, including `\va`..`\vz` / `\vA`..`\vZ` bold-vector macros and a `cond` theorem env). Bibliography is `refs.bib` with `\bibliographystyle{plain}`.
- Build with `latexmk -pdf qp1.tex` (latexmk artifacts `*.aux`, `*.bbl`, `qp1.pdf`, etc. are gitignored).
- Sections after the appendix break (`Rainfall model`, particle filter / PEnKF likelihoods, diagonal approximation, GEnKF Metropolis-within-Gibbs) hold the derivations that the code in `rainfall.py` and `enkf.py` implements. Keep code and derivations consistent when changing either. The main body sections (Simulation study, Extensions, Discussion, Conclusion) are still mostly stubs with an outline in comments.
- `references/` holds paper text/bib for cited work; not code.

## Simulation code (`code/simulation/`)

Python with PyTorch, run from `code/simulation/` (modules import each other with bare `from rainfall import *`, and notebooks load data via `./data/...`). Use the project venv: `.venv/bin/python` (Python 3.14; torch, numpy, scipy, matplotlib, xarray, jupyter). Notebooks use `%autoreload`, so editing `.py` files takes effect without restarting the kernel.

Model (everything is built on `rainfall.py`): latent theta with prior `1 + Gamma(20, 10)`; state `x` is a length-100 vector (grid on [0, 100]) with power-exponential covariance; dynamics are `x -> 0.8 x` plus spatially correlated noise (variance 0.36); observations at `m` random stations are `y = x + N(0, 0.2^2)` and `z = max(y, 0)^theta`. So `z` is zero-inflated and observation likelihoods are taken w.r.t. Lebesgue + Dirac(0). Theta is redrawn from the prior each step (`update_theta`), i.e. assumed independent over time.

Module layout and how they connect:
- `rainfall.py`: priors, dynamics, simulators, all log-likelihoods (`log_likelihood` for the exact particle filter, `penkf_log_likelihood` for the diagonal-Gaussian forecast approximation, `genkf_lik_and_grad` for theta's conditional with analytic gradient), and `simulate_data` / `save_data` / `load_data`. `data/rainfall.pt` (and the `.nc` copy) is the shared synthetic dataset every notebook loads; `T+1` entries for `theta`/`x` (index 0 is t=0, unobserved) but `T` entries for `y`/`z` (`y[t-1]` pairs with `x[t]`).
- `particle_filter.py`: bootstrap particle filter baseline (`filter`), used as the reference "truth" posterior.
- `enkf.py`: `penkf` and `genkf`. Shared pieces: `enkf_gain` / `enkf_update` (perturbed-observation EnKF update through an M x M system, never forming the n x n covariance), `conditional_ys` (draws latent `y | z, x, theta`, truncated normal for dry stations), and a Gibbs loop that alternates y-draws with EnKF updates using shuffled ancestors (systematic resampling with equal weights). They differ in how theta is handled:
  - `penkf`: a weighted set of `n` theta particles, each with its own `m`-member ensemble; weights come from `penkf_log_likelihood`, with ESS-triggered systematic resampling (threshold `0.66 n`). Returns `(particles, log_weights, esses, ensembles)`.
  - `genkf`: each ensemble member carries its own theta, which is updated inside the Gibbs loop by HMC (`hmc.py`, 1-d fixed step size) using `genkf_lik_and_grad`. Returns `(particles, log_weights)` where particles are `(theta, x)` pairs with equal weights.
- `hmc.py`: minimal 1-d HMC; rejects proposals whose log density is non-finite (this is how the `theta <= 1` prior boundary is enforced).
- `*.ipynb` (`penkf`, `genkf`, `particle_filter`, `rainfall`): driver/plotting notebooks. They seed with `torch.manual_seed(548)`, run the filter on the saved data, and plot theta/x posteriors (KDEs, random projections, credible bands) against the truth at `t = 10`.

Gotchas:
- `rainfall.py` has stub functions (`forecast_state`, `enk_update`) that are just `pass`; the real implementations are in `enkf.py`.
- `normalize`, `get_ess`/`ess` are duplicated between `enkf.py` and `particle_filter.py`; because both `from rainfall import *`, star-import order matters if you import both modules.
- Hardcoded constants that must agree across the code and the paper: `sigma = 0.2` (observation noise), forecast model noise variance `0.36` (also hardcoded in `penkf` as `Sigma + 0.36`), `T = 10` Gibbs steps.
- Regenerating the dataset overwrites `data/rainfall.pt`/`.nc`, which all notebook results (and the paper's figures) depend on.
