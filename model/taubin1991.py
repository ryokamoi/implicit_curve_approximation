import numpy as np

from utils.polynomial import Polynomial2D
from utils.params import Params


def taubin1991(dataset: np.ndarray, params: Params) -> Polynomial2D:
    """
    The classic least square method
    G. Taubin, 1991 ("Estimation of Planar Curves, Surfaces, and Nonplanar Space Curves Defined by Implicit
    Equations with Applications to Edge and Range Image Segmentation")

    Parameters
    ----------
    dataset : np.ndarray
    params : Params

    Returns
    -------
    Polynomial2D

    """

    poly = Polynomial2D(params.degree)
    M = poly.M(dataset)
    val, vec = np.linalg.eig(M.dot(M.T))
    weight = vec[:, np.argmin(val)]
    poly.set_weight(weight)

    return poly


if __name__ == "__main__":
    print("exit")
