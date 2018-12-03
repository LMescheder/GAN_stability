import numpy as np
from diracgan.util import sigmoid, clip


class VectorField(object):
    def __call__(self, theta, psi):
        theta_isfloat = isinstance(theta, float)
        psi_isfloat = isinstance(psi, float)
        if theta_isfloat:
            theta = np.array([theta])
        if psi_isfloat:
            psi = np.array([psi])

        v1, v2 = self._get_vector(theta, psi)

        if theta_isfloat:
            v1 = v1[0]
        if psi_isfloat:
            v2 = v2[0]

        return v1, v2

    def postprocess(self, theta, psi):
        theta_isfloat = isinstance(theta, float)
        psi_isfloat = isinstance(psi, float)
        if theta_isfloat:
            theta = np.array([theta])
        if psi_isfloat:
            psi = np.array([psi])
        theta, psi = self._postprocess(theta, psi)
        if theta_isfloat:
            theta = theta[0]
        if psi_isfloat:
            psi = psi[0]

        return theta, psi

    def step_sizes(self, h):
        return h, h

    def _get_vector(self, theta, psi):
        raise NotImplemented

    def _postprocess(self, theta, psi):
        return theta, psi


# GANs
def fp(x):
    return sigmoid(-x)


def fp2(x):
    return -sigmoid(-x) * sigmoid(x)


class GAN(VectorField):
    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = theta * fp(psi*theta)
        return v1, v2


class NSGAN(VectorField):
    def _get_vector(self, theta, psi):
        v1 = -psi * fp(-psi*theta)
        v2 = theta * fp(psi*theta)
        return v1, v2


class WGAN(VectorField):
    def __init__(self, clip=0.3):
        super().__init__()
        self.clip = clip

    def _get_vector(self, theta, psi):
        v1 = -psi
        v2 = theta

        return v1, v2

    def _postprocess(self, theta, psi):
        psi = clip(psi, self.clip)
        return theta, psi


class WGAN_GP(VectorField):
    def __init__(self, reg=1., target=0.3):
        super().__init__()
        self.reg = reg
        self.target = target

    def _get_vector(self, theta, psi):
        v1 = -psi
        v2 = theta - self.reg * (np.abs(psi) - self.target) * np.sign(psi)
        return v1, v2


class GAN_InstNoise(VectorField):
    def __init__(self, std=1):
        self.std = std

    def _get_vector(self, theta, psi):
        theta_eps = (
            theta + self.std*np.random.randn(*([1000] + list(theta.shape)))
        )
        x_eps = (
            self.std * np.random.randn(*([1000] + list(theta.shape)))
        )
        v1 = -psi * fp(psi*theta_eps)
        v2 = theta_eps * fp(psi*theta_eps) - x_eps * fp(-x_eps * psi)
        v1 = v1.mean(axis=0)
        v2 = v2.mean(axis=0)
        return v1, v2


class GAN_GradPenalty(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = +theta * fp(psi*theta) - self.reg * psi
        return v1, v2


class NSGAN_GradPenalty(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(-psi*theta)
        v2 = theta * fp(psi*theta) - self.reg * psi
        return v1, v2


class GAN_Consensus(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = +theta * fp(psi*theta)

        # L  0.5*(psi**2 + theta**2)*f(psi*theta)**2
        v1reg = (
            theta * fp(psi*theta)**2
            + 0.5*psi * (psi**2 + theta**2) * fp(psi*theta)*fp2(psi*theta)
        )
        v2reg = (
            psi * fp(psi*theta)**2
            + 0.5*theta * (psi**2 + theta**2) * fp(psi*theta)*fp2(psi*theta)
        )
        v1 -= self.reg * v1reg
        v2 -= self.reg * v2reg

        return v1, v2

