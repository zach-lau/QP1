""" Implementation of the PEnKF """

from rainfall import *
from hmc import hmc

def normalize(log_weights):
    """ Normalize log weights so the weights sum to 1 """
    m = max(log_weights)
    log_weights = log_weights - m
    z = torch.logsumexp(log_weights, 0)
    return log_weights - z

def get_ess(log_weights):
    """ Effective sample size of normalized log weights """
    return 1/(log_weights.exp()**2).sum()

def estimate_mean_variance_diagonal(x):
    """ diagonal estimate of mean and (unbiased) variance from ensemble"""
    X = torch.stack(x, dim=0)
    return X.mean(dim=0), X.var(dim=0)

# Ensemble update
def forecast_ensemble(xs):
    """ Propagate each member through the forecast model, without noise """
    return [model(x) for x in xs]
def noise_ensemble(xs):
    """ Add noise to existing forecast ensemble"""
    return [add_noise(x) for x in xs]

def enkf_gain(X, stations, sigma):
    """ Ensemble Kalman gain K = P H^T (H P H^T + sigma^2 I)^{-1}, shape (n, m)

    X: forecast ensemble, (M, n)
    stations: indices of the stations observed
    sigma: observation noise
    """
    M = X.shape[0]
    # H just selects stations so we never form H or the full n x n covariance
    A = X - X.mean(dim=0)                           # (M, n)
    HA = A[:, stations]                             # (M, m)
    # Should be using Q and tapering here...
    # K = A^T HA (HA^T HA/(M-1) + sigma^2 I)^{-1}/(M-1), computed through the
    # M x M system by the push-through identity
    G = sigma**2*(M-1)*torch.eye(M) + HA @ HA.T     # (M, M)
    return A.T @ torch.linalg.solve(G, HA)

def enkf_update(xs, stations, y, sigma, K=None):
    """ Perform an EnKF update with perturbed observations. The ensemble size
    is assumed to be less than the number of stations, so the gain is computed
    through an M x M system.
    
    xs: forecast ensemble (list of tensors)
    stations: indices of the stations observed
    y: observations of x at the stations, either (m,) or one row per member
    sigma: observation noise
    K: gain from enkf_gain, computed from xs if not given

    Returns: the updated ensemble (list of tensors)
    """
    X = torch.stack(xs, dim=0)                      # (M, n)
    if K is None:
        K = enkf_gain(X, stations, sigma)           # (n, m)
    # Create noised observations d
    d = y + sigma*torch.randn(X.shape[0], len(stations))   # (M, m)
    innovation = d - X[:, stations]                 # (M, m)
    return list(X + innovation @ K.T)

def conditional_ys(xs, stations, z, theta, sigma):
    """ Draw y | x, z, theta for each ensemble member, shape (M, m).
    For z > 0, y = z^{1/theta}. For z = 0, y ~ N(x, sigma^2) truncated to
    y <= 0, drawn by inverting the cdf.

    xs: ensemble (list of tensors)
    z: observations at the stations
    """
    X = torch.stack(xs, dim=0)[:, stations]         # (M, m)
    wet = z > 1e-12
    # Inverse cdf: y = x + sigma Phi^{-1}(u Phi(-x/sigma)) for u ~ U(0, 1)
    u = torch.rand_like(X)
    p = (u*torch.exp(log_cdf(-X/sigma))).clamp(min=torch.finfo(X.dtype).tiny)
    y_dry = (X + sigma*torch.special.ndtri(p)).clamp(max=0)
    return torch.where(wet, z**(1/theta), y_dry)

def conditional_update(f_ensemble, stations, z, theta, sigma):
    """
    Run a bunch of Gibbs samplers in parallel to compute the ensemble update,
    marginalizing over the unobserved y. Ancestors are picked by systematic
    resampling (a shuffle, since the weights are equal) to reduce variance.

    f_ensemble: forecast ensemble without noise (list of tensors)
    stations: indices of the stations observed
    z: observations at the stations
    theta: parameter of the observation model
    sigma: observation noise

    Returns: the updated ensemble (list of tensors)
    """

    T = 10
    prior = noise_ensemble(f_ensemble)
    xs = prior
    K = enkf_gain(torch.stack(prior, dim=0), stations, sigma)
    for _ in range(T):
        # Draw ys
        # For z = 0 draw y from a truncated normal given x, for z > 0,
        # y = z^{1/\theta}
        ys = conditional_ys(xs, stations, z, theta, sigma)
        # Update xs based on enkf update. The ancestors are picked by
        # systematic resampling, which with equal weights uses each ancestor
        # once, so it is just a shuffle
        ancestors = [prior[j] for j in torch.randperm(len(prior))]
        xs = enkf_update(ancestors, stations, ys, sigma, K)
    return xs

def systematic_resample(log_weights):
    """ Systematic resampling. Returns the indices of the ancestors, a
    LongTensor with one entry per weight, so index j appears roughly
    n*w_j times.

    log_weights: log weights, normalized or not
    """
    n = len(log_weights)
    cdf = normalize(log_weights).exp().cumsum(0)
    # One uniform draw, then n evenly spaced points on [0, 1)
    positions = (torch.rand(()) + torch.arange(n))/n
    # Guard against the cdf ending slightly below 1 from rounding
    return torch.searchsorted(cdf, positions).clamp(max=n-1)

def penkf(stations, zs, n=10, m=10, T=None):
    """ 
    stations: list of stations for observations
    zs: observations
    n: number of particles
    T: time steps to filter if None do all
    
    Returns: tuple of particles, weights, effective sample sizes, and the
    ensemble of each particle at the last time step
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
        new_weights = log_weights.clone()
        new_ensembles = []
        for i, theta in enumerate(particles):
            f_ensemble = forecast_ensemble(ensembles[i]) # no noise yet
            # Propose new particle
            theta = update_theta()
            # Update weights using diagonal approximation
            mu, Sigma = estimate_mean_variance_diagonal(f_ensemble)
            SigmaHat = Sigma + 0.36 # Hardcode variance from rainfall model
            new_weights[i] += penkf_log_likelihood(mu, SigmaHat, stations, z, theta, sigma)
            new_particles.append(theta)
            # Update ensemble using Gibbs to marginalize over unobserved y
            new_ensembles.append(conditional_update(f_ensemble, stations, z, theta, sigma)) 
        log_weights = normalize(new_weights)
        esses.append(get_ess(log_weights))
        # Resample with the updated weights when they get too uneven, after
        # which the weights are equal again
        if esses[-1] < 0.66*n:
            new_parents = systematic_resample(log_weights)
            particles = [new_particles[i] for i in new_parents]
            ensembles = [new_ensembles[i] for i in new_parents]
            log_weights = torch.full((n,), -math.log(n))
        else:
            particles = new_particles
            ensembles = new_ensembles
    return particles, log_weights, esses, ensembles

def gibbs_update(f_ensemble, stations, z, sigma):
    """ Perform gibbs update. Each member carries its own theta, started from
    the prior, and all members are updated in parallel.

    f_ensemble: forecast ensemble without noise (list of tensors)
    stations: indices of the stations observed
    z: observations at the stations
    sigma: observation noise

    Returns: tuple of the updated ensemble (list of tensors) and thetas (M,)
    """
    M = len(f_ensemble)
    prior = noise_ensemble(f_ensemble)
    T = 10 # 10 gibb's steps
    xs = prior
    thetas = torch.stack([draw_prior_theta() for _ in range(M)])
    K = enkf_gain(torch.stack(prior, dim =0), stations, sigma)
    for _ in range(T):
        # Draw new y's
        # For each element of z, if z = 0 draw from half normal
        # otherwise y = z^(1/theta)
        ys = conditional_ys(xs, stations, z, thetas[:, None], sigma)
        # Draw new x
        # Shuffle ancestors
        # Perform enkf update for x given y and theta with K above
        ancestors = [prior[j] for j in torch.randperm(M)]
        xs = enkf_update(ancestors, stations, ys, sigma, K)
        # Draw new theta from conditional distribution, with HMC started at
        # the current theta of each member
        thetas = torch.stack([
            hmc(thetas[i], lambda t, x=xs[i]: genkf_lik_and_grad(float(t), x, stations, z, sigma),
                stepsize=0.02)
            for i in range(M)])
    return list(zip(thetas, xs))

def genkf(stations, zs, n=10, T=None):
    """ 
    stations: list of stations for observations
    zs: observations
    n: number of particles
    T: time steps to filter if None do all
    
    Returns: tuple of particles, weights, effective sample sizes, and the
    ensemble of each particle at the last time step
    """
    sigma = 0.2 # known sigma for drawing y from x
    particles = []
    for _ in range(n):
        theta = draw_prior_theta()
        x = draw_prior_state()
        particles.append((theta, x))
    log_weights = torch.full((n,), -math.log(n)) # adds to 1
    for z in zs[:T]: # if T is None do all zs
        """ Implement bootstrap particle filter """
        f_ensemble = forecast_ensemble([x for (_,x) in particles]) # no noise yet
        # Propose new particle
        particles = gibbs_update(f_ensemble, stations, z, sigma)
    return particles, log_weights