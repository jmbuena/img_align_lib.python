
# @brief Optimizers Factory
# @author Jose M. Buenaposada
# @date 2017/10/11
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
#import numpy as np
#import cv2
from img_align.cost_functions import CostFunctionFactory
from img_align.optimizers import OptimizerGaussNewton


class OptimizerFactory:
    """
    The interface for the factory of Optimization algorithms.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        """
        return

    def getOptimizer(self, config):
        """
          This method returns the Optimizer object that is especified in the
          config dictionary. Raises Exceptions whith errors.

          @param config is a python dictionary with the optimization algorithm parameters.
        """

        if 'optimizer_name' not in config:
            raise LookupError('optimizer_name param missing')

        optimizer_name = config['optimizer_name']
        if (optimizer_name == '') or (optimizer_name is None):
            return ValueError('optimizer_name param is empty')

        cf_factory = CostFunctionFactory()
        cost_function = cf_factory.getCostFunction(config)

        if optimizer_name == 'GaussNewton':

            optimizer_params = dict()
            if 'max_iter' in config:
                optimizer_params['max_iter'] = config['max_iter']

            if 'tol_gradient' in config:
                optimizer_params['tol_gradient'] = config['tol_gradient']

            if 'tol_params' in config:
                optimizer_params['tol_params'] = config['tol_params']

            if 'show_iter' in config:
                optimizer_params['show_iter'] = config['show_iter']

            if 'profiling' in config:
                optimizer_params['profiling'] = config['profiling']

            return OptimizerGaussNewton(cost_function, **optimizer_params)

        return ValueError('optimizer_name value {} is not recognized'.format(optimizer_name))
