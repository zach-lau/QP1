""" Baseline for the filters: RMSPE and CRPS of the state at t = 10 when drawing
straight from the prior, ignoring all of the observations.

Run from code/simulation: .venv/bin/python prior_baseline.py [n]
"""
import sys

from rainfall import *
from metrics import record_metrics

t = 10
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000 # number of prior draws

data = load_data("./data/rainfall.pt")
x = data["x"]

torch.manual_seed(548)
X = torch.stack([draw_prior_state() for _ in range(n)]).numpy() # (n, grid)
row = record_metrics("results/prior.csv", f"prior_T{t}", X, x[t].numpy())
print(f"n = {n}: RMSPE = {row['rmspe']:.3f}, mean CRPS = {row['mean_crps']:.3f}")
