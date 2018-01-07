
# @brief Model Image Gray Factory
# @author Jose M. Buenaposada
# @date 2017/10/17
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import numpy as np
import cv2
from img_align.object_models import ModelImageGray


class ModelImageGrayFactory:
    """
    The interface for the factory of ModelImageGray target objects.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        """
        return

    def getObjectModel(self, config):
        """
          This method returns the ModelImageGray object that is especified in the
          config object. It returns None if there is an error.

          @param config is a python dictionary with the motion model parameters.
        """
        template_image = None
        template_image_shape = None
        if 'template_image' not in config:
            if 'template_image_shape' not in config:
                raise LookupError('template_image and template_image_shape params missing')
            else:
                template_image_shape = config['template_image_shape']
                template_image_shape = (int(template_image_shape[0]), int(template_image_shape[1]))

        else:
            template_image = config['template_image']
            if template_image is None:
                raise ValueError('template_image param is empty')

            if not isinstance(template_image, np.ndarray):
                template_image = cv2.imread(template_image)

        if 'template_equalize' not in config:
            template_equalize = False
        else:
            template_equalize = config['template_equalize']
            if template_equalize is None:
                template_equalize = False

        if template_image is not None:
            return ModelImageGray(template_image, equalize=template_equalize)
        else:
            return ModelImageGray(template_image_shape=template_image_shape, equalize=template_equalize)
