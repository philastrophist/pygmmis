from copy import deepcopy
from functools import partial
from itertools import cycle

import numpy as np
import pytest
from matplotlib.patches import Ellipse
from pygmmis.visual import GMMTracker
from bayesian.extreme_deconvolution.incomplete import CompleteXDGMMBase as XDGMM


if __name__ == '__main__':
    from numpy.random import RandomState
    seed = 13
    rng = RandomState(seed)
    np.random.seed(seed)

    ndim, ncomp = 2, 2
    model = XDGMM(ncomp, ndim)
    model.mu = np.array([[0., 0.], [10., 0.]])
    model.V = np.asarray([np.eye(2), np.eye(2)]) *1.
    model.alpha = np.array([1., 1.])
    model.alpha /= model.alpha.sum()

    def selection(x):
        return (x[:, 1] > -0)

    data = model.sample(5000)
    observed_data = data[selection(data)]
    model.fit(observed_data)

    from pygmmis import pygmmis

    gmm = pygmmis.GMM(K=ncomp, D=ndim)

    w = 0.1  # minimum covariance regularization, same units as data
    eta = 10  #  covariance regularization for the mean, w/eta has same units as data
    cutoff = 50  # segment the data set into neighborhood within x sigma around components
    tol = 1e-6  # tolerance on logL to terminate EM
    oversampling = 10
    maxiter = 1000


    gmm.mean = model.mu.copy() - 2.
    gmm.covar = model.V.copy()
    gmm.amp = model.alpha.copy()



    from pygmmis.backend import Backend
    backend = Backend('backend')
    logL, U = pygmmis.fit(gmm, observed_data, init_method='none', sel_callback=selection, w=w, eta=eta, cutoff=cutoff,
                          oversampling=oversampling, tol=tol, rng=rng, maxiter=maxiter, backend=backend)

    tracker = GMMTracker(backend, data)
    tracker.figures()
    tracker.plot_trace(0, 12)