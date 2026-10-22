""" Implementation of the PEnKF """

from rainfall import *

def normalize(log_weights):
    # Sum of weights should be 1
    m = max(log_weights)
    log_weights = log_weights - m
    z = torch.logsumexp(log_weights, 0)
    return log_weights - z

def ess(log_weights):
    # Return the ess
    return 1/(log_weights.exp()**2).sum()

def filter(stations, zs, n=10, m=10, T=None):
    """ 
    stations: list of stations for observations
    zs: observations
    n: number of particles
    T: time steps to filter if None do all
    
    Returns: tuple of particles and weights at last timem step
    """
    sigma = 0.2 # known sigma for drawing y from x
    particles = []
    ensembles = []
    esses = []
    for _ in range(n):
        theta = draw_prior_theta()
        ensembles.append([])
        for _ in range(m):
            x = draw_prior_state()
            ensembles[-1].append(x)
        particles.append(theta)
    log_weights = torch.full((n,), -math.log(n)) # adds to 1
    for z in zs[:T]: # if T is None do all zs
        """ Implement bootstrap particle filter """
        new_particles = []
        new_weights = log_weights
        for i, theta in enumerate(particles):
            this_ensemble = ensembles[i]
            # Propose new particle
            theta = update_theta()
            # Update weights
            new_weights[i] += log_likelihood(x, stations, z, theta, sigma)
            new_particles.append(theta)
        particles = new_particles
        log_weights = normalize(new_weights)
        esses.append(ess(log_weights))
    return particles, log_weights, esses