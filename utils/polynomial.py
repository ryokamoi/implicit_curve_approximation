from typing import List, Tuple

import numpy as np


class Polynomial1D(object):
    def __init__(self, degree: int) -> None:
        self.weight = None
        self.degree = int(degree)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert self.weight is not None, "weight is not specified"
        original_shape = np.shape(x)
        x = np.reshape(x, -1)
        m = self.M(x)
        return np.reshape(self.weight.dot(m), original_shape)

    def set_weight(self, weight: np.ndarray) -> None:
        assert np.shape(weight)[0] == self.degree + 1
        self.weight = np.array(weight)

    def M(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x : np.ndarray
            size [num of datapoints]

        Returns
        -------
        M: np.ndarray
            size [degree+1, num of datapoints]

        """

        output = np.ones(np.shape(x)[0])
        for xn in range(self.degree + 1):
            m = np.power(x, xn)
            output = np.vstack([output, m])
        return output[1:]


class Polynomial2D(object):
    def __init__(self, degree: int) -> None:
        self.weight = None
        self.degree = int(degree)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self.weight is not None, "weight is not specified"
        original_shape = np.shape(x)
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        datapoints = np.vstack([x, y]).T
        m = self.M(datapoints)
        return np.reshape(self.weight.dot(m), original_shape)

    def set_weight(self, weight: np.ndarray) -> None:
        assert np.shape(weight)[0] == (self.degree+1) * (self.degree+2) / 2
        self.weight = weight

    def degrees(self, d: int) -> List[Tuple[int, int]]:
        x = []
        y = []
        for yd in range(d+1):
            for xd in range(d+1-yd):
                x.append(xd)
                y.append(yd)
        return list(zip(x, y))

    def M(self, datapoints: np.ndarray) -> None:
        """

        Parameters
        ----------
        datapoints : np.ndarray
            size [2, num of datapoints]

        Returns
        -------
        M : np.ndarray
            size [(d+1)(d+2)/2, num of datapoints]

        """

        dt = np.array(datapoints, np.float32).T
        x = dt[0]
        y = dt[1]
        output = np.ones(np.shape(datapoints)[0])
        degrees = self.degrees(self.degree)
        for xn, yn in degrees:
            m = np.power(x, xn) * np.power(y, yn)
            output = np.vstack([output, m])
        return output[1:]

    def nablaM(self, datapoints: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        datapoints : np.ndarray
            size [2, num of datapoints]

        Returns
        -------
        nablaM : np.ndarray
            size [(d+1)(d+2)/2, 2, num of datapoints]

        """

        dt = np.array(datapoints, np.float32).T
        x = dt[0]
        y = dt[1]
        output = np.ones([1, 2, np.shape(datapoints)[0]])
        degrees = self.degrees(self.degree)
        for xn, yn in degrees:
            if xn > 0:
                dmdx = xn * np.power(x, xn-1) * np.power(y, yn)
            else:
                dmdx = np.zeros(np.shape(x))
            if yn > 0:
                dmdy = yn * np.power(x, xn) * np.power(y, yn-1)
            else:
                dmdy = np.zeros(np.shape(y))
            output = np.vstack([output, np.expand_dims(np.vstack([dmdx, dmdy]), axis=0)])
        return output[1:]


if __name__ == "__main__":
    p2_1d = Polynomial1D(2)
    datapoints = np.array([1, 2, 3, 4])
    M1 = p2_1d.M(datapoints)
    p2_1d.set_weight(np.ones(3))
    print(p2_1d(datapoints))

    p2 = Polynomial2D(2)
    datapoints = np.array([[1, 2], [3, 4], [1, 5]])
    M2 = p2.M(datapoints)
    nablaM = p2.nablaM(datapoints)
    p2.set_weight(np.ones(int((2+1) * (2+2) / 2)))
    print("exit")
