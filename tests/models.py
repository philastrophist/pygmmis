from functools import partial
from itertools import cycle

import numpy as np
from astroML.density_estimation import XDGMM
import pytest
from matplotlib.patches import Ellipse


def no_selection(x):
    """x is an array of shape == (dims, components, samples)"""
    return x[0] > -np.inf


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

    ndim, ncomp = 2, 4
    model, selection = gaussians_in_a_box(ndim, ncomp, 1, 5)

    data = model.sample(5000)
    observed_data = data[selection(data)]


    plt.scatter(*data.T, c='k', s=1)
    plt.scatter(*observed_data.T, c='g', s=1)

    def plot_ellipse(ax, mu, covariance, color, linewidth=2, alpha=0.5):
        var, U = np.linalg.eig(covariance)
        angle = 180. / np.pi * np.arccos(np.abs(U[0, 0]))
        e = Ellipse(mu, 2 * np.sqrt(5.991 * var[0]),
                    2 * np.sqrt(5.991 * var[1]),
                    angle=angle)
        e.set_alpha(alpha)
        e.set_linewidth(linewidth)
        e.set_edgecolor(color)
        e.set_facecolor(color)
        e.set_fill(False)
        ax.add_artist(e)

        return e

    for i in range(model.n_components):
        plot_ellipse(plt.gca(), model.mu[i], model.V[i], 'b')


    import pygmmis
    # more sophisticated option: use the covariance of the nearest neighbor.
    def covar_tree_cb(coords, tree, covar):
        """Return the covariance of the nearest neighbor of coords in data."""
        dist, ind = tree.query(coords, k=1)
        return covar[ind.flatten()]


    from sklearn.neighbors import KDTree
    covar = np.asarray([np.eye(ndim) for _ in data]) * 0.1
    tree = KDTree(data, leaf_size=100)
    covar = pygmmis.createShared(covar)
    covar_cb = partial(covar_tree_cb, tree=tree, covar=covar)



    gmm = pygmmis.GMM(K=ncomp, D=ndim)

    w = 0.1  # minimum covariance regularization, same units as data
    cutoff = 0.01  # segment the data set into neighborhood within 5 sigma around components
    tol = 1e-3  # tolerance on logL to terminate EM

    # run EM
    import logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logL, U = pygmmis.fit(gmm, data, init_method='kmeans', sel_callback=selection, w=w, cutoff=cutoff,
                          tol=tol, rng=rng, split_n_merge=gmm.K * (gmm.K - 1) * (gmm.K - 2) / 2, covar_callback=covar_cb)


    for i in range(gmm.K):
        plot_ellipse(plt.gca(), gmm.mean[i], gmm.covar[i], 'k')

