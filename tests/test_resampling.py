from copy import deepcopy

import pygmmis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm
import datetime
from functools import partial
import logging

def plotResults(orig, data, gmm, patch=None, description=None, disp=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(orig[:,0], orig[:,1], 'o', mfc='None', mec='r', mew=1)
    missing = np.isnan(data)
    if missing.any():
        data_ = data.copy()
        data_[missing] = -5 # put at limits of plotting range
    else:
        data_ = data
    ax.plot(data_[:,0], data_[:,1], 's', mfc='b', mec='None')#, mew=1)

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # compute sum_k(p_k(x)) for all x
    p = gmm(coords).reshape((B,B))
    # for better visibility use arcshinh stretch
    p = np.arcsinh(p/1e-4)
    cs = ax.contourf(p, 10, extent=(-5,15,-5,15), cmap=plt.cm.Greys)
    for c in cs.collections:
        c.set_edgecolor(c.get_facecolor())

    # plot boundary
    if patch is not None:
        import copy
        if hasattr(patch, '__iter__'):
            for p in patch:
                ax.add_artist(copy.copy(p))
        else:
            ax.add_artist(copy.copy(patch))

    # add description and complete data logL to plot
    logL = gmm(orig, as_log=True).mean()
    if description is not None:
        ax.text(0.05, 0.95, r'%s' % description, ha='left', va='top', transform=ax.transAxes, fontsize=20)
        ax.text(0.05, 0.89, '$\log{\mathcal{L}} = %.3f$' % logL, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    else:
        ax.text(0.05, 0.95, '$\log{\mathcal{L}} = %.3f$' % logL, ha='left', va='top', transform=ax.transAxes, fontsize=20)

    # show size of error dispersion as Circle
    if disp is not None:

        circ1 = patches.Circle((12.5, -2.5), radius=disp, fc='b', ec='None', alpha=0.5)
        circ2 = patches.Circle((12.5, -2.5), radius=2*disp, fc='b', ec='None', alpha=0.3)
        circ3 = patches.Circle((12.5, -2.5), radius=3*disp, fc='b', ec='None', alpha=0.1)
        ax.add_artist(circ1)
        ax.add_artist(circ2)
        ax.add_artist(circ3)
        ax.text(12.5, -2.5, r'$\sigma$', color='w', fontsize=20, ha='center', va='center')

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
    fig.show()

def plotDifferences(orig, data, gmms, avg, l, patch=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    #ax.plot(orig[:,0], orig[:,1], 'o', mfc='None', mec='r', mew=1)
    ax.plot(data[:,0], data[:,1], 's', mfc='b', mec='None')#, mew=1)

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # compute sum_k(p_k(x)) for all x
    pw = avg(coords).reshape((B,B))

    # use each run and compute weighted std
    p = np.empty((T,B,B))
    for r in range(T):
        # compute sum_k(p_k(x)) for all x
        p[r,:,:] = gmms[r](coords).reshape((B,B))

    p = ((p-pw[None,:,:])**2 * l[:,None, None]).sum(axis=0)
    V1 = l.sum()
    V2 = (l**2).sum()
    p /= (V1 - V2/V1)

    p = np.arcsinh(np.sqrt(p)/1e-4)
    cs = ax.contourf(p, 10, extent=(-5,15,-5,15), cmap=plt.cm.Greys, vmin=np.arcsinh(pw/1e-4).min(), vmax=np.arcsinh(pw/1e-4).max())
    for c in cs.collections:
        c.set_edgecolor(c.get_facecolor())

    # plot boundary
    if patch is not None:
        import copy
        if hasattr(patch, '__iter__'):
            for p in patch:
                ax.add_artist(copy.copy(p))
        else:
            ax.add_artist(copy.copy(patch))

    ax.text(0.05, 0.95, 'Dispersion', ha='left', va='top', transform=ax.transAxes, fontsize=20)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
    fig.show()

def getBox(coords):
    box_limits = np.array([[0,0],[10,10]])
    return (coords[:,0] > box_limits[0,0]) & (coords[:,0] < box_limits[1,0]) & (coords[:,1] > box_limits[0,1]) & (coords[:,1] < box_limits[1,1])

def getHole(coords):
    x,y,r = 6.5, 6., 2
    return ((coords[:,0] - x)**2 + (coords[:,1] - y)**2 > r**2)

def getBoxWithHole(coords):
    return getBox(coords)*getHole(coords)

def getHalfDensity(coords, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < 0.5] = 0
    return mask

def getTaperedDensity(coords, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < coords[:,0]/8] = 0
    return mask

def getCut(coords):
    return (coords[:,0] < 6)

def getAll(coords):
    return np.ones(len(coords), dtype='bool')

def getSelection(type="hole", rng=np.random):
    if type == "hole":
        cb = getHole
        ps = patches.Circle([6.5, 6.], radius=2, fc="none", ec='k', lw=1, ls='dashed')
    if type == "box":
        cb = getBox
        ps = patches.Rectangle([0,0], 10, 10, fc="none", ec='k', lw=1, ls='dashed')
    if type == "boxWithHole":
        cb = getBoxWithHole
        ps = [patches.Circle([6.5, 6.], radius=2, fc="none", ec='k', lw=1, ls='dashed'),
            patches.Rectangle([0,0], 10, 10, fc="none", ec='k', lw=1, ls='dashed')]
    if type == "cut":
        cb = getCut
        ps = lines.Line2D([6, 6],[-5, 15], ls='dotted', lw=1, color='k')
    if type == "tapered":
        cb = partial(getTaperedDensity, rng=rng)
        ps = lines.Line2D([8, 8],[-5, 15], ls='dotted', lw=1, color='k')
    if type == "under":
        cb = getUnder
        ps = None
    if type == "over":
        cb = getOver
        ps = None
    if type == "none":
        cb = getAll
        ps = None
    return cb, ps



if __name__ == '__main__':

    # set up test
    N = 400             # number of samples
    K = 3               # number of components
    T = 1               # number of runs
    sel_type = "box"    # type of selection
    disp = 0.5          # additive noise dispersion
    w = 0.1             # minimum covariance regularization [data units]
    cutoff = 5          # cutoff distance between components [sigma]
    seed = 8365         # seed value
    oversampling = 10   # for missing data: imputation samples per observed sample
    # show EM iteration results
    logging.basicConfig(format='%(message)s',level=logging.INFO)

    # define RNG for run
    from numpy.random import RandomState
    rng = RandomState(seed)

    # draw N points from 3-component GMM
    D = 2
    gmm = pygmmis.GMM(K=3, D=2)
    gmm.amp[:] = np.array([ 0.36060026,  0.27986906,  0.206774])
    gmm.amp /= gmm.amp.sum()
    gmm.mean[:,:] = np.array([[ 0.08016886,  0.21300697],
                              [ 0.70306351,  0.6709532 ],
                              [ 0.01087670,  0.852077]])*10
    gmm.covar[:,:,:] = np.array([[[ 0.08530014, -0.00314178],
                                  [-0.00314178,  0.00541106]],
                                 [[ 0.03053402, 0.0125736],
                                  [0.0125736,  0.01075791]],
                                 [[ 0.00258605,  0.00409287],
                                 [ 0.00409287,  0.01065186]]])*100

    # data come from pure GMM model or one with background?
    orig = gmm.draw(N, rng=rng)

    # add isotropic errors on data
    noisy = orig + rng.normal(0, scale=disp, size=(len(orig), D))

    # get observational selection function
    cb, ps = getSelection(sel_type, rng=rng)

    # apply selection
    sel = cb(noisy)
    data = noisy[sel]
    # single covariance for all samples
    covar = disp**2 * np.eye(D)

    start_gmm = deepcopy(gmm)
    start_gmm.mean = np.array([[2.61641184, 6.87213943],
                               [3.37617126, 4.40660136],
                               [7.99196769, 7.07302165]])
    start_gmm.covar = np.array([[[11.55256116, -7.49213141],
                                [-7.49213141, 14.33994686]],
                               [[10.04469829, -2.12086539],
                                [-2.12086539,  9.18871938]],
                               [[ 6.87497209,  9.23363409],
                                [ 9.23363409, 17.46852466]]])
    start_gmm.amp = np.array([0.19484221, 0.57429356, 0.23086422])
    plotResults(orig, data, start_gmm, patch=ps, description="starting point")

    covar_cb = partial(pygmmis.covar_callback_default, default=np.eye(D)*disp**2)
    start = datetime.datetime.now()
    rng = RandomState(seed)
    analytical_gmm = deepcopy(start_gmm)
    pygmmis.fit(analytical_gmm, data, covar=covar, maxiter=100,
                w=w, cutoff=cutoff, init_method='none', rng=rng, tol=1e-10)
    print ("execution time %ds" % (datetime.datetime.now() - start).seconds)
    plotResults(orig, data, analytical_gmm, patch=ps, description="$\mathtt{GMMis}$ - no resampling")


    start = datetime.datetime.now()
    rng = RandomState(seed)
    resampled_gmm = deepcopy(start_gmm)

    class MyTransform(pygmmis.Transform):
        def forward(self, x):
            return x #- 100

        def backward(self, x):
            return x# + 100

    transform = MyTransform()
    observed_data = transform.backward(data)



    pygmmis.fit(resampled_gmm, observed_data, covar=covar, maxiter=100, n_resamples=50, transform=pygmmis.Transform(),
                w=w, cutoff=cutoff, init_method='none', rng=rng, tol=1e-10)
    print ("execution time %ds" % (datetime.datetime.now() - start).seconds)
    plotResults(orig, data, resampled_gmm, patch=ps, description="$\mathtt{GMMis}$ - resampled")