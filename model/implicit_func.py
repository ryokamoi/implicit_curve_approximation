import numpy as np

from utils.polynomial import Polynomial1D


class IncreasingPolynomial(object):
    def __init__(self, degree: int) -> None:
        assert degree == 5, "not implemented for specified degree"
        self.degree = degree
        self.weight = None
        self.polynomial = Polynomial1D(degree)

    def __call__(self, datapoints: np.ndarray):
        self.polynomial.set_weight(self.weight)
        return self.polynomial(datapoints)

    def set_weight(self, coeffs: np.ndarray) -> None:
        assert np.shape(coeffs)[0] == 7, "size of weight is invalid"
        c0, a0, b0, a1, b1, a2, b2 = coeffs
        c5 = a2**2 / 5 + b2**2 / 5
        c4 = a1*a2/2 + b1*b2/2
        c3 = (2.0/3)*a0*a2 + a1**2/3 + (2.0/3)*b0*b2 + b1**2/3
        c2 = a0*a1 + b0*b1
        c1 = a0**2 + b0**2
        self.weight = np.array([c0, c1, c2, c3, c4, c5])


class HomeomorphicPolynomial(object):
    def __init__(self, degree:int, output_type=1, increasingfunc=IncreasingPolynomial) -> None:
        assert degree % 2 == 1, "degree should be odd"
        self.degree = degree
        self.f1 = increasingfunc(self.degree)
        self.f2 = increasingfunc(self.degree)
        self.g1 = increasingfunc(self.degree)
        self.g2 = increasingfunc(self.degree)

        assert output_type in [1, 2], "invalid old_output type"
        self.output_type = output_type

    def __call__(self, x: np.ndarray, y: np.ndarray):
        f1_val = self.f1(x)
        f2_val = self.f2(y)
        g1_val = self.g1(x)
        g2_val = self.g2(y)

        if self.output_type == 1:
            return f1_val+f2_val, g1_val-g2_val
        elif self.output_type == 2:
            return g1_val-g2_val, f1_val+f2_val
        else:
            raise ValueError("Invalid old_output type")

    def set_weight(self, weight: np.ndarray):
        if isinstance(self.f1, IncreasingPolynomial):
            assert np.sum(np.shape(weight)) == 4*7, "invalid number of weights"
        weight = np.reshape(weight, [4, -1])
        self.f1.set_weight(weight[0])
        self.f2.set_weight(weight[1])
        self.g1.set_weight(weight[2])
        self.g2.set_weight(weight[3])


class UnitCircleImplicit(object):
    def __init__(self, homeomorphism: HomeomorphicPolynomial) -> None:
        self.homeomorphism = homeomorphism

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        original_shape = np.shape(x)
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        h1, h2 = self.homeomorphism(x, y)
        return np.reshape(np.sqrt(np.square(h1) + np.square(h2)) - 1, original_shape)


class FaithfulImplicit(object):
    def __init__(self, degree:int, threeL=False, insensitive=0.0,
                 homeomorphism_type=1, increasingfunc=IncreasingPolynomial) -> None:
        if isinstance(increasingfunc, IncreasingPolynomial):
            assert degree == 5, "not implemented for specified degree"
        self.degree = degree

        assert homeomorphism_type in [1, 2], "invalid homeomorphism_type"
        self.homeomorphism = HomeomorphicPolynomial(self.degree, output_type=homeomorphism_type,
                                                    increasingfunc=increasingfunc)

        self.threeL = threeL
        self.epsilon = 0.5

        self.zeroset = None
        self.outer = None
        self.inner = None

        self.insensitive = insensitive # epsilon for insensitive loss function

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        original_shape = np.shape(x)
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        h1, h2 = self.homeomorphism(x, y)
        return np.reshape(np.sqrt(np.square(h1) + np.square(h2)) - 1, original_shape)

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        original_shape = np.shape(x)
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        h1, h2 = self.homeomorphism(x, y)
        return np.reshape(np.square(h1) + np.square(h2) - 1, original_shape)

    def set_weight(self, weight: np.ndarray) -> None:
        if isinstance(self.homeomorphism.f1, IncreasingPolynomial):
            assert np.sum(np.shape(weight)) == 4*7, "invalid number of weights"
        self.homeomorphism.set_weight(weight)

    def set_training_data(self, zeroset: np.ndarray, outer=None, inner=None) -> None:
        assert np.shape(zeroset)[1] == 2, "invalid shape of datapoints. the shape should be [n, 2]."
        self.zeroset = zeroset
        if self.threeL:
            self.outer = outer
            self.inner = inner

    def epsilon_insensitive_square(self, loss: np.ndarray, insensitive_e=0.1) -> np.ndarray:
        return np.sum(np.maximum(0.0, np.square(loss) - insensitive_e))

    def loss_func(self, weight: np.ndarray) -> np.ndarray:
        assert self.zeroset is not None, "training data should be specified by self.set_training_data"
        if self.threeL:
            assert self.outer is not None, "training data should be specified by self.set_training_data"
            assert self.inner is not None, "training data should be specified by self.set_training_data"
        assert np.sum(np.shape(weight)) == 4*7, "invalid number of weights"
        self.set_weight(weight)
        data_t = np.array(self.zeroset).T
        x = data_t[0]
        y = data_t[1]
        zeroloss = self.loss(x, y)
        loss = self.epsilon_insensitive_square(zeroloss, self.insensitive)
        if self.threeL:
            data_t = np.array(self.outer).T
            x = data_t[0]
            y = data_t[1]
            outerloss = self.loss(x, y) - self.epsilon
            loss += self.epsilon_insensitive_square(outerloss, self.insensitive)

            data_t = np.array(self.inner).T
            x = data_t[0]
            y = data_t[1]
            innerloss = self.loss(x, y) + self.epsilon
            loss += self.epsilon_insensitive_square(innerloss, self.insensitive)
        return loss
