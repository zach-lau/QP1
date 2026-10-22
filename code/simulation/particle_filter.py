""" Implement particle filter for the rainfall model """
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

def filter(stations, zs, n=100, T=None):
    """ 
    stations: list of stations for observations
    zs: observations
    n: number of particles
    T: time steps to filter if None do all
    
    Returns: tuple of particles and weights at last timem step
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
    return particles, log_weights, esses