""" Replicate the simulation study over independent simulated datasets.

For each seed, simulate a dataset (T = 10, same settings as data/rainfall.pt),
run the prior baseline and every filter on all ten time points and on the final
time point only, and record RMSPE and mean CRPS of the state at t = 10. Also
record the ESS of the particle filter and PEnKF at every step, with and without
resampling.

Each seed writes its own files in results/replicates/ (so seeds can run in
parallel, and finished seeds are skipped on a restart). Use summarize_replicates.py
to aggregate.

Run from code/simulation: .venv/bin/python replicate.py FIRST_SEED LAST_SEED
"""
import csv
import os
import sys
import time

import numpy as np

from rainfall import *
from metrics import rmspe, mean_crps
import particle_filter
import enkf

torch.set_num_threads(1) # seeds run in parallel as separate processes
torch.set_grad_enabled(False) # nothing here uses autograd, HMC takes analytic gradients

T = 10
OUT = "results/replicates"

def score(method, experiment, X, x_true, weights=None):
    return {"method": method, "experiment": experiment,
            "rmspe": rmspe(X, x_true, weights), "mean_crps": mean_crps(X, x_true, weights)}

def penkf_samples(particles, log_weights, ensembles):
    """ Posterior samples (n, m, grid) of a PEnKF run and their flat weights """
    X = torch.stack([torch.stack(e) for e in ensembles]).numpy()
    return X, np.repeat(log_weights.exp().numpy()/X.shape[1], X.shape[1])

def run_seed(seed):
    torch.manual_seed(seed)
    data = simulate_data(T=T)
    stations = data[1][1]
    z = torch.stack([d[4] for d in data[1:]]) # (T, m)
    x_true = data[T][2].numpy()

    def seeded(offset):
        torch.manual_seed(10_000*offset + seed) # each run reproducible on its own

    metrics, esses = [], []
    def record_esses(method, experiment, es):
        esses.extend({"method": method, "experiment": experiment, "t": t, "ess": float(e)}
                     for t, e in enumerate(es, start=1))

    seeded(1)
    X = torch.stack([draw_prior_state() for _ in range(1000)]).numpy()
    metrics.append(score("prior", "prior", X, x_true))

    for experiment, zs in [("filtering", z), ("single_update", z[T-1:T])]:
        seeded(2)
        particles, log_weights, es = particle_filter.filter(stations, zs, n=1000, T=T)
        X = torch.stack([p[1] for p in particles]).numpy()
        metrics.append(score("particle_filter", experiment, X, x_true, log_weights.exp().numpy()))
        record_esses("particle_filter", experiment, es)

        seeded(3)
        particles, log_weights, es, ensembles = enkf.penkf(stations, zs, n=100, m=10, T=T)
        X, w = penkf_samples(particles, log_weights, ensembles)
        metrics.append(score("penkf", experiment, X, x_true, w))
        record_esses("penkf", experiment, es)

        seeded(4)
        particles, _ = enkf.genkf(stations, zs, n=100, T=T)
        X = torch.stack([p[1] for p in particles]).numpy()
        metrics.append(score("genkf", experiment, X, x_true))

    # Without resampling, only to look at the ESS
    seeded(5)
    _, _, es = particle_filter.filter(stations, z, n=1000, T=T, resample=False)
    record_esses("particle_filter", "filtering_no_resampling", es)
    seeded(6)
    _, _, es, _ = enkf.penkf(stations, z, n=100, m=10, T=T, resample=False)
    record_esses("penkf", "filtering_no_resampling", es)

    return metrics, esses

def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    for seed in range(first, last + 1):
        done = f"{OUT}/metrics_seed{seed}.csv"
        if os.path.exists(done):
            continue
        start = time.time()
        metrics, esses = run_seed(seed)
        write(f"{OUT}/esses_seed{seed}.csv", [{"seed": seed, **r} for r in esses])
        write(done, [{"seed": seed, **r} for r in metrics]) # last, marks the seed as done
        print(f"seed {seed} done in {time.time() - start:.0f}s", flush=True)
