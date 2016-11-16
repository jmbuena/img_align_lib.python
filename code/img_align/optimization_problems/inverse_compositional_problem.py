
# @brief Optimization problem interface for 'inverse compositional solutions'
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import numpy as np
from img_align.optimization_problems import OptimizationProblem


class InverseCompositionalProblem(OptimizationProblem):

    """
    The interface for the Optimization Problems in Inverse Compositional solutions

    The inverse compositional based problems assume that the optimization
    problem jacobian J, is constant.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model):
        """

        :param object_model:
        :param motion_model:
        :return:
        """
        super(InverseCompositionalProblem, self).__init__(object_model, motion_model)

        self.__initialized = False
        self.__J = None
        self.__invJ = None
        self.__setupMatrices()

        return

    def computeJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """

        if not self.__initialized:
            self.__setupMatrices()

        return self.__J

    def computeInverseJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        if not self.__initialized:
            self.__setupMatrices()

        return self.__invJ

    def __setupMatrices(self):
        self.__J = self.computeConstantJacobian()
        self.__invJ = np.linalg.pinv(self.__J)

        self.__initialized = True
        return

    @abc.abstractmethod
    def computeConstantJacobian(self):
        return
