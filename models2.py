from functools import partial
from itertools import cycle

import numpy as np
from astroML.density_estimation import XDGMM
import pytest
from matplotlib.patches import Ellipse

from pygmmis.visual import GMMTracker


def no_selection(x):
    """x is an array of shape == (samples, dims)"""
    return x[:, 0] > -np.inf


def gaussians_in_a_line(ndim, ncomp):
    xdgmm = XDGMM(ncomp)
    xdgmm.mu = np.array([[i]*ndim for i in range(ncomp)])
    xdgmm.V = np.asarray([np.eye(ndim)]*ncomp) * 0.15

    c = cycle([0.25, 0.5])
    xdgmm.alpha = np.array([next(c) for _ in range(ncomp)])
    xdgmm.alpha /= xdgmm.alpha.sum()

    def gaussians_in_a_line_selection(x):
        return (x[:, 1] < x[:, 0] + 0.5) & (x[:, 1] > x[:, 0] - 1)

    return xdgmm, gaussians_in_a_line_selection


def gaussians_in_a_box(ndim, ncomp, density, percent=5):
    xdgmm = XDGMM(ncomp)
    xdgmm.V = np.asarray([np.eye(ndim)] * ncomp) * 0.10
    width = np.asarray([[2 * np.sqrt(5.991 * var) for var in np.linalg.eig(v)[0]] for v in xdgmm.V]).max()
    xdgmm.mu = np.random.uniform(-width / density, width / density, size=(ncomp, ndim))

    c = cycle([0.25, 0.5, 0.1])
    xdgmm.alpha = np.array([next(c) for _ in range(ncomp)])
    xdgmm.alpha /= xdgmm.alpha.sum()

    np.random.seed(1)
    data = xdgmm.sample(10000)
    limits = np.percentile(data, [percent, 100-percent], axis=0)

    def gaussians_in_a_box_selection(x):
        return ((x > limits[0]) & (x < limits[1])).all(axis=1)

    return xdgmm, gaussians_in_a_box_selection


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # define RNG for deterministic behavior
    from numpy.random import RandomState
    seed = 13
    rng = RandomState(seed)
    np.random.seed(seed)

    ndim, ncomp = 2, 3
    model, _ = gaussians_in_a_box(ndim, ncomp, 0.75, 5)

    def selection(x):
        return (x[:, 1] > -1.5) & (x[:, 1] < 2.5) & (x[:, 0] > 0.3) &  (x[:, 0] < 2.75)


    data = model.sample(5000)
    observed_data = data[selection(data)]

    from pygmmis import pygmmis

    gmm = pygmmis.GMM(K=ncomp, D=ndim)

    w = 0.1  # minimum covariance regularization, same units as data
    eta = 10  #  covariance regularization for the mean, w/eta has same units as data
    cutoff = 50  # segment the data set into neighborhood within x sigma around components
    tol = 1e-6  # tolerance on logL to terminate EM
    oversampling = 10
    maxiter = 500

    # run EM
    import logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)



    # logL, U = pygmmis.fit(gmm, data, init_method='kmeans', w=w, cutoff=cutoff, tol=tol, rng=rng, maxiter=1,
    #                       split_n_merge=gmm.K * (gmm.K - 1) * (gmm.K - 2) / 2)

    # gmm.mean = np.array([[1.60750837, 0.04992414],
    #                    [1.18327347, 0.21677725],
    #                    [1.37233111, 0.13234453],
    #                    [1.95724086, 0.1714379 ]])
    gmm.mean = model.mu.copy()
    # gmm.mean[3] += [0, 1]
    # gmm.mean[0] += 1
    # gmm.mean[1] -= 1
    gmm.covar = model.V.copy()
    gmm.amp = model.alpha.copy()


    from pygmmis.backend import Backend
    backend = Backend('backend')

    logL, U = pygmmis.fit(gmm, observed_data, init_method='none', sel_callback=selection, w=w, eta=eta, cutoff=cutoff,
                          oversampling=oversampling,
                          tol=tol, rng=rng, maxiter=1000, split_n_merge=10,
                          backend=backend)
    #
    #
    # gmm.mean[3] = observed_data.mean(axis=0)
    # gmm.covar[3] = np.eye(2) * 0.1
    #
    # gmm.mean[2] = observed_data.mean(axis=0)
    # gmm.covar[2] = np.eye(2) * 0.1
    #
    # gmm.mean[0] = observed_data.mean(axis=0)
    # gmm.covar[0] = np.eye(2) * 0.1
    #
    # logL, U = pygmmis.fit(gmm, observed_data, init_method='none', sel_callback=selection, w=w, eta=eta, cutoff=cutoff,
    #                       oversampling=oversampling,
    #                       tol=tol, rng=rng, maxiter=maxiter, frozen=[1],
    #                       backend=backend)

    # plt.plot(backend.master.get_values('log_L'))
    # tracker = GMMTracker(backend, observed_data)
    # tracker.figures()
    # tracker.plot_trace()