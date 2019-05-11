import numpy as np

from utils.polynomial import Polynomial2D
from utils.params import Params


def local_tangent(dataset: np.ndarray) -> np.ndarray:
    """
    simple local tangent for ordered dataset

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    local tangent : ndarray with size [num of datapoints, 2]

    """

    dataset = np.array(dataset)
    pre = np.roll(dataset, 1, axis=0)
    next = np.roll(dataset, -1, axis=0)
    diff = next - pre
    output = diff / np.reshape(np.sqrt(np.sum(np.square(diff), axis=1)), [-1, 1])
    return output


def tasdizen2000(dataset: np.ndarray, params: Params) -> Polynomial2D:
    """
    The Gradient-One method
    T. Tasdizen, et al., 2000 ("Improving the Stability of Algebraic Curves for Applications")

    Parameters
    ----------
    dataset : np.ndarray
    params : Params

    Returns
    -------
    Polynomial2D

    """

    Tan = local_tangent(dataset)
    Norm = np.roll(Tan, 1, axis=1)
    Norm[:, 0] *= -1

    poly = Polynomial2D(params.degree)
    M = poly.M(dataset)
    nablaM = poly.nablaM(dataset)

    S = M.dot(M.T)
    nablaM = np.transpose(nablaM, axes=[2, 0, 1])
    Snsq = np.reshape(np.matmul(nablaM, np.expand_dims(Norm, axis=-1)), [np.shape(dataset)[0], -1])
    Sn = Snsq.T.dot(Snsq)
    Stsq = np.reshape(np.matmul(nablaM, np.expand_dims(Tan, axis=-1)), [np.shape(dataset)[0], -1])
    St = Stsq.T.dot(Stsq)
    Gn = np.sum(np.reshape(np.matmul(nablaM, np.expand_dims(Norm, axis=-1)), [np.shape(dataset)[0], -1]), axis=0)

    weight = params.mu * np.linalg.inv((S + params.mu * (Sn+St))).dot(Gn)
    poly.set_weight(weight)

    return poly


if __name__ == "__main__":
    dataset = [[0, 0], [0, 1], [1, 2]]
    lt = local_tangent(np.array(dataset))
    print("exit")
