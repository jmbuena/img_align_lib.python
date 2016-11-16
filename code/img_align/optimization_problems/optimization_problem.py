
# @brief Optimization problem interface
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class OptimizationProblem:
    """
    The interface for the OptimizationProblem to be used with Optimizer
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model, show_debug_info=False):
        """

        :param object_model:
        :param motion_model:
        :return:
        """
        self.__object_model = object_model
        self.__motion_model = motion_model
        self.show_debug_info = show_debug_info
        return

    @abc.abstractmethod
    def computeCostFunction(self,
                            residual_vector,
                            motion_params,
                            delta_params,
                            image):
        """

        :param residual_vector:
        :param motion_params:
        :param delta_params:
        :param image:
        :return:
        """
        return

    @abc.abstractmethod
    def computeResidual(self, image, motion_params, show_debug_info=False):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def computeJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def computeInverseJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def updateMotionParams(self, motion_params, inc_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return

    def getObjectModel(self):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return self.__object_model

    def getMotionModel(self):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return self.__motion_model

  
