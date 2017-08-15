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

    # @abc.abstractmethod
    # def warpImage(self, image, motion_params, template_coords):
    #     """
    #
    #     :param image:
    #     :param motion_params:
    #     :param template_coords:
    #     :return:
    #     """
    #     return

    @abc.abstractmethod
    def getNumParams(self):
        return

    def validParams(self, motion_params):
        return True
