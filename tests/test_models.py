import numpy as np
from .models import no_selection, gaussians_in_a_box, gaussians_in_a_line

import pytest

def test_correct_answer(model_function, ndim, ncomp, selection, mutol, Vtol, alphatol):
    model, selection_function = model_function(ndim, ncomp)
    samples = model.sample(10000)

    if selection:
        truncated_samples = samples[selection_function(samples.T[:, None, :])[0]] #samples[selection_function(samples.T)]
    else:
        truncated_samples = samples
        selection_function = no_selection

    xdgmm = IncompleteXDGMM(3, 2, selection_function, verbose=True)  # 13

    xdgmm.initialise(truncated_samples, 1)
    xdgmm.fit(truncated_samples, max_iter=1000, tol=1e-6, random_state=0)


    orders = list(permutations(range(xdgmm.n_components)))
    diffs = [np.sum([(xdgmm.mu[j] - model.mu[i]) ** 2 for i,j in enumerate(order)]) for order in orders]
    order = list(orders[np.argmin(diffs)])


    np.testing.assert_allclose(xdgmm.mu[order], model.mu, atol=mutol)
    np.testing.assert_allclose(xdgmm.V[order], model.V, atol=Vtol)
    np.testing.assert_allclose(xdgmm.alpha[order], model.alpha, atol=alphatol)