import os
from typing import Union

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model.implicit_func import FaithfulImplicit
from utils.polynomial import Polynomial2D


def read_dataset(filename: str) -> np.ndarray:
    """

    Parameters
    ----------
    filename : str

    Returns
    -------
    np.ndarray
        size [2, num of datapoints]

    """

    dataset = []
    with open(filename, "r") as f:
        for l in f.readlines():
            line = l[:-1]
            x, y = map(float, line.split())
            dataset.append([x, y])
    return np.array(dataset, np.float32)


def visualize_implicit(func: Union[Polynomial2D, FaithfulImplicit], filename="../old_output/sample.png",
                       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                       epsilon=0.01, delta=0.01, dataset=None) -> None:
    x = np.arange(xlim[0], xlim[1], delta)
    y = np.arange(ylim[0], ylim[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    image = np.ones(np.shape(Z))
    image[np.where(Z>epsilon)] = 0
    image[np.where(Z<-epsilon)] = 0

    if dataset is not None:
        dx = np.array(dataset).T[0]
        dy = np.array(dataset).T[1]

    extentlist = [xlim[0], xlim[1], ylim[0], ylim[1]]
    fig = plt.imshow(np.zeros(np.shape(Z)), cmap=plt.cm.binary, extent=extentlist)
    if dataset is not None:
        plt.plot(dx, dy, ".", color="r")
    cset = plt.contour(Z, [0], linewidths=2, cmap=plt.cm.gist_heat, extent=extentlist, zorder=3)
    binary, _ = os.path.splitext(filename)
    binary += "_bi.png"
    plt.savefig(binary)

    fig = plt.imshow(Z, cmap=plt.cm.RdBu, extent=extentlist)
    if dataset is not None:
        plt.plot(dx, dy, ".", color="r")
    cset = plt.contour(Z, np.arange(-2, 2, 0.05), linewidths=1, cmap=plt.cm.Set2, extent=extentlist, zorder=3)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    color, _ = os.path.splitext(filename)
    color += "_col.png"
    plt.savefig(color)


def unitcirc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.power(x, 2) + np.power(y, 2) - 1


if __name__ == "__main__":
    visualize_implicit(unitcirc, filename="../old_output/unitcirc.png")
    print("exit")
