
# @brief Motion Model Factory
# @author Jose M. Buenaposada
# @date 2017/10/10
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
from img_align.motion_models import MotionHomography8P


class MotionModelFactory:
    """
    The interface for the factory of Motion Models. To be used within a CostFunction.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        """
        return

    def getMotionModel(self, config):
        """
          This method returns the MotionModel object that is especified in the
          config object. It returns None if there is an error.

          @param config is a python dictionary with the motion model parameters.
        """

        if 'motion_model_name' not in config:
            raise LookupError('motion_model_name param missing')

        motion_model_name = config['motion_model_name']
        if (motion_model_name == '') or (motion_model_name is None):
            return ValueError('motion_model_name param is empty')

        if motion_model_name == 'Homography8P':
            return MotionHomography8P()

        return ValueError('motion_model_name value {} is not recognized'.format(motion_model_name))
