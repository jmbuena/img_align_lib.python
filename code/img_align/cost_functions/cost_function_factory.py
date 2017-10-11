
# @brief Cost Function Factory
# @author Jose M. Buenaposada
# @date 2017/10/04
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import img_align.motion_models
import img_align.object_models

class CostFunctionFactory:
    """
    The interface for the factory of Cost Functions. To be used with an Optimizer.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        """
        return

    def getCostFunction(self, config):
        """
          This method returns the CostFunction object that is especified in the+
          config object. It returns None if there is an error.

          @param config is a python dictionary with the motion model parameters.
        """

        if 'cost_function_name' not in config:
            raise LookupError('cost_function_name param missing')

        cost_function_name = config['cost_function_name']
        if (cost_function_name == '') or (cost_function_name is None):
            return ValueError('cost_function_name param is empty')

        object_model = ObjectModelFactory.getObjectModel(config)
        motion_model = MotionModelFactory.getMotionModel(config)

        if cost_function_name == 'L2ImagesInvComp':
            return CostFunL2ImagesInvComp(object_model, motion_model, show_debug_info=False)

        raise ValueError('cost_function_name value {} is not recognized'.format(cost_function_name))
