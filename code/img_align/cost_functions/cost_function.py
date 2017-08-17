
# @brief Cost Function interface
# @author Jose M. Buenaposada
# @date 2017/08/16
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class CostFunction:
    """
    The interface for the CostFunction to be used with an Optimizer
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, show_debug_info=False):
        """
        :show_debug_info:
        """
        self.show_debug_info = show_debug_info
        return
