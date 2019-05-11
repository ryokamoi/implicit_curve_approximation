import numpy as np
import scipy
from scipy import optimize

from model.implicit_func import FaithfulImplicit
from utils.params import Params


def keren2004(zeroset: np.ndarray, params: Params, outer=None, inner=None, iter=10) -> FaithfulImplicit:
    """
    Topologically Faithful Fitting
    D. Keren, 2004 "Topologically Faithful Fitting of Simple Closed Curves"
    In this code, Naive Nelder-Mead method (from scipy) is used.

    Parameters
    ----------
    zeroset : np.ndarray
    params : class Params
    outer : np.ndarray
        only used for 3L algorithm
    inner : np.ndarray
        only used for 3L algorithm
    iter : int
        iteration for Nelder-Mead method

    Returns
    -------

    """

    poly = FaithfulImplicit(params.degree, threeL=params.threeL)
    if poly.threeL:
        poly.set_training_data(zeroset, outer, inner)
    else:
        poly.set_training_data(zeroset)
    minloss = 1e+9
    weight = None
    for _ in range(iter):
        init_weight = np.random.normal(scale=4, size=28)
        result = scipy.optimize.minimize(poly.loss_func, init_weight, method="Nelder-Mead")
        if result.fun < minloss:
            weight = result.x
            minloss = result.fun
    poly.set_weight(weight)
    return poly

if __name__ == "__main__":
    print("exit")
