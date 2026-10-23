"""
Generate synthetic rainfall data in accordance with Katzfuss et al. 2020
"""
import torch
import math
import xarray as xr

# Model priors and dynamics
def draw_prior_theta():
    """ Draw from prior theta"""
    return 1 + torch.distributions.Gamma(20, 10).sample()
def log_prior(theta):
    """ Log prior density of theta, where theta - 1 ~ Gamma(20, 10) """
    theta = torch.as_tensor(theta, dtype=torch.float32)
    return torch.distributions.Gamma(20., 10.).log_prob(theta - 1)
def grad_log_prior(theta):
    """ Gradient of the log prior density of theta wrt theta, where
    theta - 1 ~ Gamma(20, 10): (a-1)/(theta-1) - b """
    a, b = 20., 10.
    return (a-1)/(theta-1) - b
def draw_prior_state(mean = 0.2, scale = 10, power=1.8):
    # Draw prior state based on power exponential covariance function
    x = torch.linspace(0, 100, 100)
    d = (x[:,None] - x[None,:]).abs()
    R = torch.exp(-(d/scale)**power)
    mu = mean*torch.ones(x.shape[0])
    # Small jitter for numerical stability of the Cholesky factorization
    R = R + 1e-6*torch.eye(x.shape[0])
    return torch.distributions.MultivariateNormal(mu, covariance_matrix=R).sample()
def draw_y(x, indices, sigma=0.2):
    """ Draw y conditional on x """
    obs = x[indices]
    return obs + sigma*torch.randn_like(obs)
def draw_z(y, theta):
    return y.clamp(min=0)**theta
def model(x):
    return x*0.8
def add_noise(x):
    """ Add spatially correlated, zero-mean noise with variance 0.36 """
    return x + 0.6*draw_prior_state(mean=0.0)
def update_state(x):
    """ Could incorporate prior over noise here too"""
    # Make the state updates also spatially correlated
    return add_noise(model(x))
def update_theta():
    return draw_prior_theta() # assume independence

# GEnKF
def forecast_state(x):
    """ Forecast state. No theta because we have forecast independence"""
    pass
def enk_update(x, y, theta):
    """Perform enk update for x based on y and theta"""
    pass

def norm_log_pdf(x, mu=0, sigma=1):
    # Returns log norm pdf up to a constant
    return -(x-mu)**2/2/sigma**2
def log_cdf(x):
    """ Log of the standard normal cdf, stable in the lower tail """
    return torch.special.log_ndtr(x)

# Likelihoods
def log_likelihood(x, stations, z, theta, sigma):
    """ Log likelihood with fixed sigma, wrt lebesgue + dirac(0), up to a
    constant. x is the full state, stations the indices observed in z"""
    eps = 1e-12
    obs = x[stations]
    zpos = z[z > eps]
    xpos = obs[z > eps]
    xzero = obs[~(z > eps)]
    m = len(zpos)
    return norm_log_pdf(zpos**(1/theta), xpos, sigma).sum() \
        + (1/theta-1)*torch.log(zpos).sum() \
        - m*math.log(theta*sigma) \
        + log_cdf(-xzero/sigma).sum()

def penkf_log_likelihood(mu, Sigma, stations, z, theta, sigma):
    """ Log marginal likelihood p(z | theta) under the diagonal approximation
    of the PEnKF forecast, wrt lebesgue + dirac(0), up to the same constant as
    log_likelihood. mu is the forecast mean for the full state and Sigma the
    diagonal of the forecast covariance (a vector of variances, one per grid
    cell), which does not include the observation noise; that is added here
    as sigma**2. """
    eps = 1e-12
    obs_mu = mu[stations]
    obs_var = Sigma[stations]
    # Observation noise adds to the forecast variance
    s2 = sigma**2 + obs_var
    pos = z > eps
    zpos, mupos, s2pos = z[pos], obs_mu[pos], s2[pos]
    muzero, s2zero = obs_mu[~pos], s2[~pos]
    m = len(zpos)
    # (1/theta) z^(1/theta-1) Norm(z^(1/theta); mu, s2) for z > 0
    # Phi(-mu / s) for z = 0
    return norm_log_pdf(zpos**(1/theta), mupos, s2pos.sqrt()).sum() \
        - 0.5*torch.log(s2pos).sum() \
        + (1/theta-1)*torch.log(zpos).sum() \
        - m*math.log(theta) \
        + log_cdf(-muzero/s2zero.sqrt()).sum()

def genkf_lik_and_grad(theta, x, stations, z, sigma=0.2):
    """ Log of the GEnKF conditional for theta, log p(theta) + log L(theta),
    up to a constant, and its gradient wrt theta. Only z is needed from the
    observations, since y enters through z = y^theta. Returns (ell, grad);
    outside the prior support (theta <= 1) returns (-inf, 0). """
    if theta <= 1:
        return torch.tensor(-math.inf), torch.tensor(0.)
    eps = 1e-12
    pos = z > eps
    zpos, xpos = z[pos], x[stations][pos]
    logz = torch.log(zpos)
    zt = zpos**(1/theta)
    ell = log_prior(theta) + log_likelihood(x, stations, z, theta, sigma)
    grad = grad_log_prior(theta) \
        + (-1/theta - logz/theta**2 + logz*zt*(zt-xpos)/(sigma**2*theta**2)).sum()
    return ell, grad

# Simulate data
def simulate_data(n = 100, m = 75, T = 1):
    """ Generate data, assuming a well specified prior and model. Returns
    a list of tuples (theta, x, y, z). The first element of the list 
    is at t=0, which doesn't contain any observations."""
    theta = draw_prior_theta()
    x = draw_prior_state()
    stations = torch.randperm(n)[:m].sort().values
    ans = [(theta, None, x, None, None)]
    for t in range(T):
        x = update_state(x)
        theta = update_theta()
        y = draw_y(x, stations)
        z = draw_z(y, theta)
        ans.append((theta, stations, x, y , z))
    return ans

# Saving
def save_data(data, path, **meta):
    """ Save simulated data as both <path>.pt (torch) and <path>.nc (NetCDF).
    theta and x have T+1 entries (include t=0); y and z have T, so y[t-1] goes
    with x[t]. meta (seed, n, m, T, ...) is stored alongside."""

    theta = torch.stack([torch.as_tensor(d[0]) for d in data])  # (T+1,)
    x = torch.stack([d[2] for d in data])                       # (T+1, n)
    stations = data[1][1]                                       # (m,)
    y = torch.stack([d[3] for d in data[1:]])                   # (T, m)
    z = torch.stack([d[4] for d in data[1:]])                   # (T, m)
    torch.save({"theta": theta, "x": x, "stations": stations,
                "y": y, "z": z, "meta": meta}, f"{path}.pt")
    T, n = len(data) - 1, x.shape[1]
    xr.Dataset(
        {"theta": (("time0",), theta.numpy()),
         "x": (("time0", "grid"), x.numpy()),
         "y": (("time", "station"), y.numpy()),
         "z": (("time", "station"), z.numpy())},
        coords={"time0": range(T + 1), "time": range(1, T + 1),
                "grid": range(n), "station": stations.numpy()},
        attrs=meta,
    ).to_netcdf(f"{path}.nc")

def load_data(path):
    """ Load the dict saved by save_data from <path> """
    return torch.load(f"{path}")
