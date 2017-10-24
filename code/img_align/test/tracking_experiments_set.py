# @brief Optimization algorithm interface.
# @author Jose M. Buenaposada
# @date 2017/10/10
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import numpy as np
import img_align.object_models
import img_align.motion_models
import img_align.cost_functions
import img_align.optimizers
from img_align.test import TrackingExperimentPlanar


class TrackingExperimentsSet:

    def __init__(self):

        self.__tracking_experiments = []
        return

    def add(self, track_exp):
        '''
        :param track_exp: TrackingExperiment
        '''

        self.__tracking_experiments.append(track_exp)

    def run(self):
        '''
        Run all the visual tracking experiments
        '''

        for exp in self.__tracking_experiments:
            exp.run()

        return

    def evaluateResults(self, gt_dir, results_dir, evaluation_dir):
        '''
        Compare the results of different algorithms
        '''

        return
