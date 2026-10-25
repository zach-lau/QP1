""" Aggregate the per-seed results of replicate.py into
results/replicates_summary.csv (mean, standard error and 95% confidence interval
half-width over seeds, from the t distribution, of RMSPE and
mean CRPS for each method and experiment), results/replicates_paired.csv (mean and
standard error over seeds of the paired difference in RMSPE and mean CRPS between
methods, which share each seed's dataset; negative means the first method is
better) and results/replicates_esses_summary.csv (mean and standard error over
seeds of the ESS at each time step), and results/replicates_esses_average.csv (the same
for the ESS averaged over the time steps within each seed).

Run from code/simulation: .venv/bin/python summarize_replicates.py
"""
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import t as t_dist

DIR = "results/replicates"

def read(pattern):
    rows = []
    for path in sorted(glob.glob(f"{DIR}/{pattern}")):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows

def mean_se(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1)/np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return values.mean(), se

def mean_se_ci(values):
    """ Mean, standard error, and half-width of the 95% t confidence interval """
    mean, se = mean_se(values)
    return mean, se, t_dist.ppf(0.975, len(values) - 1)*se if len(values) > 1 else float("nan")

PAIRS = [("genkf", "penkf"), ("genkf", "particle_filter"), ("penkf", "particle_filter")]

def paired(rows, path):
    """ Paired differences between methods within each seed and experiment """
    by = {(r["seed"], r["experiment"], r["method"]): r for r in rows}
    out = []
    for experiment in sorted({r["experiment"] for r in rows} - {"prior"}):
        seeds = sorted({r["seed"] for r in rows if r["experiment"] == experiment}, key=int)
        for a, b in PAIRS:
            for metric in ("rmspe", "mean_crps"):
                d = np.array([float(by[(s, experiment, a)][metric]) - float(by[(s, experiment, b)][metric])
                              for s in seeds])
                mean, se, ci = mean_se_ci(d)
                out.append([experiment, f"{a} - {b}", metric, len(d), mean, se, ci, (d < 0).mean()])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "comparison", "metric", "n_seeds", "mean_diff", "se", "ci95", "frac_first_better"])
        writer.writerows(out)
    return out

def average_ess(rows, path):
    """ Mean and standard error over seeds of the ESS averaged over time steps """
    per_seed = defaultdict(list)
    for r in rows:
        per_seed[(r["method"], r["experiment"], r["seed"])].append(float(r["ess"]))
    groups = defaultdict(list)
    for (method, experiment, _), es in per_seed.items():
        groups[(method, experiment)].append(np.mean(es))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "experiment", "n_seeds", "ess_mean", "ess_se", "ess_ci95"])
        for (method, experiment), v in groups.items():
            writer.writerow([method, experiment, len(v), *mean_se_ci(v)])

def summarize(rows, key, values, path):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for v in values:
            groups[tuple(r[k] for k in key)][v].append(float(r[v]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(key) + ["n_seeds"] + [f"{v}_{s}" for v in values for s in ("mean", "se", "ci95")])
        for k, vs in groups.items():
            stats = [mean_se_ci(vs[v]) for v in values]
            writer.writerow(list(k) + [len(vs[values[0]])] + [x for s in stats for x in s])
    return groups

if __name__ == "__main__":
    metrics = read("metrics_seed*.csv")
    groups = summarize(metrics, ("method", "experiment"), ("rmspe", "mean_crps"),
                       "results/replicates_summary.csv")
    for r in paired(metrics, "results/replicates_paired.csv"):
        print(*r[:4], "diff %.3f (%.3f)  first better on %.0f%%" % (r[4], r[5], 100*r[7]),
              " z = %.1f" % (r[4]/r[5]))
    esses = read("esses_seed*.csv")
    summarize(esses, ("method", "experiment", "t"), ("ess",), "results/replicates_esses_summary.csv")
    average_ess(esses, "results/replicates_esses_average.csv")
    for k, vs in groups.items():
        print(*k, f"n={len(vs['rmspe'])}",
              "RMSPE %.3f (%.3f)" % mean_se(vs["rmspe"]),
              "CRPS %.3f (%.3f)" % mean_se(vs["mean_crps"]))
