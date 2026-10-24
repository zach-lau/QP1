""" Implement particle filter for the rainfall model """
from rainfall import *
from enkf import systematic_resample

def normalize(log_weights):
    # Sum of weights should be 1
    m = max(log_weights)
    log_weights = log_weights - m
    z = torch.logsumexp(log_weights, 0)
    return log_weights - z

def ess(log_weights):
    # Return the ess
    return 1/(log_weights.exp()**2).sum()

def filter(stations, zs, n=100, T=None, resample=True):
    """ 
    stations: list of stations for observations
    zs: observations
    n: number of particles
    T: time steps to filter if None do all
    resample: resample when the ess drops below 0.66 n

    Returns: tuple of particles, weights at last time step, and effective
    sample sizes (before any resampling)
    """
    sigma = 0.2 # known sigma for drawing y from x
    particles = []
    esses = []
    for i in range(n):
        theta = draw_prior_theta()
        x = draw_prior_state()
        particles.append((theta, x))
    log_weights = torch.full((n,), -math.log(n)) # adds to 1
    for z in zs[:T]: # if T is None do all zs
        """ Implement bootstrap particle filter """
        new_particles = []
        new_weights = log_weights
        for i, (theta, x) in enumerate(particles):
            # Propose new particle
            theta = update_theta()
            x = update_state(x)
            # Update weights
            new_weights[i] += log_likelihood(x, stations, z, theta, sigma)
            new_particles.append((theta, x))
        particles = new_particles
        log_weights = normalize(new_weights)
        esses.append(ess(log_weights))
        # Resample with the updated weights when they get too uneven, after
        # which the weights are equal again
        if esses[-1] < 0.66*n and resample:
            parents = systematic_resample(log_weights)
            particles = [particles[i] for i in parents]
            log_weights = torch.full((n,), -math.log(n))
    return particles, log_weights, esses