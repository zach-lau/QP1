"""Accuracy metrics for posterior samples of the state, and a recorder for the
effective sample sizes of the filters.

Same conventions as plots.py: samples are an array `X` of shape (N, grid), or
(n, m, grid) for the PEnKF where each of the n particles carries an ensemble of
m members. Optional `weights` are per-sample (flattened) weights summing to 1;
without them, samples are equally weighted. `x_true` is the true state as a
numpy array of length grid.
"""
import csv
import os

import numpy as np


def _flatten(X):
    return X.reshape(-1, X.shape[-1])


def rmspe(X, x_true, weights=None):
    """ Root mean squared prediction error of the posterior mean over the grid """
    samples = _flatten(X)
    post_mean = samples.mean(axis=0) if weights is None else weights @ samples
    return float(np.sqrt(np.mean((post_mean - x_true)**2)))


def crps(X, x_true, weights=None):
    """ CRPS of the (weighted) sample distribution at each grid point, shape (grid,).

    Uses CRPS = E|X - x| - E|X - X'|/2, where the second term is computed from
    the sorted samples as sum_i F_i (1 - F_i) (s_{i+1} - s_i) with F_i the
    weighted cdf after the i-th sample. """
    samples = _flatten(X)
    N = samples.shape[0]
    if weights is None:
        weights = np.full(N, 1/N)
    expect1 = weights @ np.abs(samples - x_true)

    order = np.argsort(samples, axis=0)
    s = np.take_along_axis(samples, order, axis=0)
    cum = np.cumsum(weights[order], axis=0)[:-1]
    expect2 = np.sum(cum*(1 - cum)*np.diff(s, axis=0), axis=0)
    return expect1 - expect2


def mean_crps(X, x_true, weights=None):
    """ CRPS averaged over the grid """
    return float(crps(X, x_true, weights).mean())


def _upsert(path, experiment, new_rows):
    """ Write new_rows (dicts) to the csv at `path`, replacing any existing rows of
    the experiment """
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["experiment"] != experiment]
    rows += new_rows
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_metrics(path, experiment, X, x_true, weights=None):
    """ Write the RMSPE and mean CRPS of one experiment to the csv at `path`,
    replacing the experiment's row if it is already there. """
    row = {"experiment": experiment, "rmspe": rmspe(X, x_true, weights),
           "mean_crps": mean_crps(X, x_true, weights)}
    _upsert(path, experiment, [row])
    return row


def record_esses(path, experiment, esses):
    """ Write the effective sample size at each time step (t = 1, ..., T) of one
    experiment to the csv at `path`, replacing the experiment's rows if they are
    already there. """
    rows = [{"experiment": experiment, "t": t, "ess": float(e)}
            for t, e in enumerate(esses, start=1)]
    _upsert(path, experiment, rows)
    return rows
