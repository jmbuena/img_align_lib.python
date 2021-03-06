
# @brief motion model (function) in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2017/08/14 (Modified)
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc


class MotionModel:
    """
    A class that defines the interface for the motion model of the target.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def scaleParams(self, motion_params, scale):
        """

        :param motion_params: current params
        :param scale: scalar value than changes the resolution of the image
        :return: np array with the updated motion params to the new scale
        """
        return

    @abc.abstractmethod
    def invMap(self, coords, motion_params):
        """
        Using the motion_params, transform the input coords from image
        coordinates to the corresponding template coordinates.

        :param coords: current coords in input image reference system
        :param motion_params: current motion params from template to image
        :return: np array with the updated coords.
        """
        return

    @abc.abstractmethod
    def map(self, coords, motion_params):
        """

        :param coords: current coords in template reference system
        :param motion_params: current motion params from template to image
        :return: np array with the updated coords
        """
        return

    @abc.abstractmethod
    def getCompositionParams(self, motion_params1, motion_params2):
        """
        Let f(x, p) the motion model function, with motion parameters p, that
        transforms coordinates x. This method computes the motion model params such as:
            f(x, composition_params) =  f(f(x, motion_params1), motion_params2)

        :param motion_params1: motion params for f(x, motion_params1).
        :param motion_params2: motion params for second application of f.
        :return: np array with the updated motion parameters.
        """
        return

    @abc.abstractmethod
    def getCompositionWithInverseParams(self, motion_params1, motion_params2):
        """
        Let f(x, p) the motion model function, with motion parameters p, that
        transforms coordinates x. This method computes the motion model params such as:
            f(x, composition_params) =  f(f^{-1}(x, motion_params2), motion_params1)

        :param motion_params1: motion params for second application of f.
        :param motion_params2: motion params for f^{-1}(x, motion_params1).
        :return: np array with the updated motion parameters.
        """
        return

    @abc.abstractmethod
    def computeJacobian(self, coords, motion_params):
        """
        Let f(x, p) the motion model function, with motion parameters p, that
        transforms coordinates x. This method computes the first derivative for the motion
        model, at given coordinates and with given motion params.

        The function computes the jacobian for N points of dimension d.

        :param coords: current coords in template reference system (Nxd)
        :param motion_params: current motion params from template to image (kx1)
        :return: np array with the jacobians (dxkxN)
        """
        return

    @abc.abstractmethod
    def computeParams(self, points_orig, points_dest):
        """
        :param points_orig
        :param points_dest
        :return the motion params that maps points_orig into points_dest:
        """

    @abc.abstractmethod
    def getIdentityParams(self):
        """
        Get the motion params that maps coords in the same coords.

            motion_model.map(coords, motion_params) returns coords

        :return the identity parameter numpy array:
        """
        return

    @abc.abstractmethod
    def getNumParams(self):
        return

    def validParams(self, motion_params):
        return True

    @abc.abstractmethod
    def generateRandomParamsIncrements(self, num_samples, n_sigmas=1):
        """
        Generates num_samples small increments motion params vectors at random. The random values
        are generated with an uniform distribution between -n_sigmas*sigma and +n_sigmas*sigma, being
        sigma the base standard deviation for that motion parameter.

        :param num_samples:
        :return: A num_params x num_samples numpy array with the random params by columns
        """
        return