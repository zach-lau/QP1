""" Implementatin of fixed step size HMC """
import math
import torch

def hmc_step(x0, eval_grad, stepsize, steps):
    """ One HMC transition (x0 if the proposal is rejected) """
    x = torch.as_tensor(x0, dtype=torch.float32)
    f, g = eval_grad(x)
    p = torch.randn(())
    h0 = -f + p**2/2
    # Leapfrog
    xn, fn, gn, pn = x, f, g, p
    pn = pn + stepsize*gn/2
    for i in range(steps):
        xn = xn + stepsize*pn
        fn, gn = eval_grad(xn)
        if not math.isfinite(fn):
            return x # left the support, reject
        if i < steps - 1:
            pn = pn + stepsize*gn
    pn = pn + stepsize*gn/2
    h1 = -fn + pn**2/2
    # Metropolis accept step, momentum flip is irrelevant as p is symmetric
    if torch.log(torch.rand(())) < h0 - h1:
        return xn
    return x

def hmc(x0, eval_grad, stepsize=0.1, steps=20, iterations=5):
    """
    1d HMC with unit mass, returning the location after `iterations` transitions.
    x0: start location
    eval_grad: function that returns log density evaluation and gradient
    stepsize, steps: leapfrog step size and number of steps per transition"""
    x = x0
    for _ in range(iterations):
        x = hmc_step(x, eval_grad, stepsize, steps)
    return x
