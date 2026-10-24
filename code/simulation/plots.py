"""Plotting helpers shared by the notebooks.

Posterior samples of the state are an array `X` of shape (N, grid), or
(n, m, grid) for the PEnKF where each of the n particles carries an ensemble of
m members. Optional `weights` are per-sample (flattened) weights summing to 1;
without them, samples are equally weighted. Every plot function takes an
optional `save` path to also write the figure to. `x_true` is the true state at time
`t` as a numpy array of length grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


def _show(save):
    """ Save the current figure to `save` if given, then show it """
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


def _flatten(X):
    return X.reshape(-1, X.shape[-1])


def _kde_grid(samples, true_value, size=500):
    return np.linspace(min(samples.min(), true_value) - 0.5,
                       max(samples.max(), true_value) + 0.5, size)


def weighted_quantiles(samples, weights, qs):
    """ Weighted quantiles of each column of samples (N, g), shape (len(qs), g) """
    order = np.argsort(samples, axis=0)
    s = np.take_along_axis(samples, order, axis=0)
    w = weights[order]
    cdf = np.cumsum(w, axis=0) - w/2 # midpoint cdf, so a single sample is not a step
    return np.stack([np.interp(qs, cdf[:, j], s[:, j]) for j in range(s.shape[1])], axis=1)


def plot_kde(samples, true_value, xlabel, true_label, weights=None, save=None):
    """ KDE of a scalar's posterior samples against its true value """
    kde = gaussian_kde(samples, weights=weights)
    grid = _kde_grid(samples, true_value)
    plt.plot(grid, kde(grid), label="KDE" if weights is None else "Weighted KDE")
    plt.axvline(true_value, color="k", linestyle="--", label=true_label)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    _show(save)


def plot_theta(thetas, true_theta, t, weights=None, save=None):
    plot_kde(thetas, true_theta, r"$\theta$", rf"True $\theta_{{{t}}}$", weights, save)


def plot_state_marginal(X, x_true, t, k, weights=None, save=None):
    """ KDE of the state at grid point k """
    label = rf"$x_{{{t},{k}}}$"
    plot_kde(_flatten(X)[:, k], x_true[k], label, f"True {label}", weights, save)


def plot_projection(X, x_true, t, w, weights=None, ensembles=None, particle_weights=None,
                    min_weight=0.01, save=None):
    """ KDE of the projection w^T x. For the PEnKF, passing the per-particle
    `ensembles` and `particle_weights` also draws each heavy particle's
    contribution to the total KDE. """
    proj = _flatten(X) @ w # (N,)
    true_proj = float(x_true @ w)

    kde = gaussian_kde(proj, weights=weights)
    grid = _kde_grid(proj, true_proj)
    plt.plot(grid, kde(grid), color="C0", linewidth=2,
             label="KDE" if weights is None else "Weighted KDE")

    if ensembles is not None:
        h = np.sqrt(kde.covariance[0, 0]) # kernel width of the total KDE
        proj = proj.reshape(len(ensembles), -1) # (n, m)

        # After resampling, copies of the same ancestor share one ensemble, so
        # group them and add up their weights
        groups = {}
        for i, e in enumerate(ensembles):
            groups.setdefault(id(e), []).append(i)

        for j, g in enumerate(g for g in groups.values()
                              if particle_weights[g].sum() > min_weight*(1 + 1e-3)): # not a tie at 1/n
            # KDE of this ensemble alone with the same kernel, scaled by its
            # weight, so the contributions of all particles add up to the total
            contribution = (particle_weights[g].sum()
                            * norm.pdf(grid[:, None], proj[g[0]], h).mean(axis=1))
            plt.plot(grid, contribution, color="C1", linewidth=1, alpha=0.7,
                     label=f"Particles with weight > {min_weight}" if j == 0 else None)

    plt.axvline(true_proj, color="k", linestyle="--", label=rf"True $w^\top x_{{{t}}}$")
    plt.xlabel(rf"$w^\top x_{{{t}}}$")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    _show(save)


def plot_state_band(X, x_true, t, stations, weights=None, save=None):
    """ Posterior mean and 95% credible band of the state over the grid """
    samples = _flatten(X)
    location = np.linspace(0, 100, samples.shape[1]) # grid used by draw_prior_state
    if weights is None:
        post_mean = samples.mean(axis=0)
        lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    else:
        post_mean = weights @ samples
        lower, upper = weighted_quantiles(samples, weights, [0.025, 0.975])

    plt.plot(location, post_mean, label="Posterior mean")
    plt.fill_between(location, lower, upper, alpha=0.3, label="95% credible interval")
    plt.plot(location, x_true, "k--", label=rf"True $x_{{{t}}}$")
    plt.plot(location[stations], np.full(len(stations), lower.min()), "|", color="gray",
             label="Stations")
    plt.xlabel("Location")
    plt.ylabel("x")
    plt.legend()
    _show(save)


def plot_heatmap(M, xlabel, title, cbar_label, save=None):
    """ Heatmap of a (T, g) array over time, with time 1..T on the y-axis """
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="Blues",
                   extent=[0, M.shape[1], 1, M.shape[0] + 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    _show(save)


def plot_theta_trace(thetas, save=None):
    """ Traceplot of theta over time """
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(range(len(thetas)), thetas, lw=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(r"Traceplot of $\theta$")
    _show(save)
