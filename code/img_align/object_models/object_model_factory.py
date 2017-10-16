
# @brief Object Model Factory
# @author Jose M. Buenaposada
# @date 2017/10/10
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import numpy as np
import cv2
import img_align.object_models


class ObjectModelFactory:
    """
    The interface for the factory of Object Models. To be used within a CostFunction.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        """
        return

    def getObjectModel(self, config):
        """
          This method returns the ObjectModel object that is especified in the
          config object. It returns None if there is an error.

          @param config is a python dictionary with the motion model parameters.
        """

        if 'object_model_name' not in config:
            raise LookupError('object_model_name param missing')

        object_model_name = config['object_model_name']
        if (object_model_name == '') or (object_model_name is None):
            return ValueError('object_model_name param is empty')

        if object_model_name == 'ImageGray':

            if 'template_image' not in config:
                raise LookupError('template_image param missing')

            template_image = config['template_image']
            if template_image is None:
                raise ValueError('template_image param is empty')

            if not isinstance(template_image, np.array):
                template_image = cv2.imread(template_image)

            if 'template_equalize' not in config:
                template_equalize = False
            else:
                template_equalize = config['template_equalize']
                if template_equalize is None:
                    template_equalize = False

            return ModelImageGray(template_image, equalize=template_equalize)

        return ValueError('object_model_name value {} is not recognized'.format(object_model_name))
