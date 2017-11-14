__author__ = 'Herakles II'

# Imports
import numpy
import copy

# OU-Process
def ornstein_uhlenbeck_1d(num_steps, num_sims, kappa, sigma, mu,
                          dist_type='arithmetic', seed=None):
    # Simulation of one dimensional Ornstein-Uhlenbeck process. We use antithetic sampling
    # Only arithmetic simulation is implemented so far!
    #
    # INPUT
    # =====
    # num_steps - natural number
    # num_sims  - even number
    # x0        - real number
    # kappa     - either real number or vector of length numSteps
    #           > mean reversion rate
    # sigma     - either real number or vector of length numSteps
    #           > model volatility
    # mu        - either real number or vector of length numSteps
    #           > mean reversion level
    # dist_type - 'arithmetic' (default) or 'geometric'
    #           - type of used distribution
    #
    # OUTPUT
    # ======
    # paths     - numpy.array of size (num_steps+1, num_sims)

    # resize kappa, sigma and mu so vector
    kappa = kappa*numpy.ones(num_steps)
    sigma = sigma*numpy.ones(num_steps)
    mu = mu*numpy.ones(num_steps)
    
    std = numpy.sqrt(sigma)

    paths_null_mr = numpy.zeros((num_steps, num_sims), dtype = numpy.float_)

    if seed is not None:
        numpy.random.seed(seed)
    increments = numpy.random.normal(size = (num_steps-1, num_sims/2))
    increments = numpy.column_stack((increments, -increments))

    for step in range(1, num_steps):
        paths_null_mr[step, :] = (1-kappa[step])*paths_null_mr[step-1, :] + increments[step-1, :]*std[step]

    paths = paths_null_mr+mu[:, numpy.newaxis]
    return paths