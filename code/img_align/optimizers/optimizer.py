
# @brief Optimization algorithm interface.
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class Optimizer:

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def solve(self, frame, former_params):
        """
        This function returns the new motion params from the image to process
        and the motion params.

        :param frame: OpenCV numpy array image
        :param former_params: motion params from the last frame call to solve
        :return: numpy array column vector that represent the new motion params
        """
        return


