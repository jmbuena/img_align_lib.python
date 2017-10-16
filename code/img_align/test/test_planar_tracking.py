# @brief Cost Function interface
# @author Jose M. Buenaposada
# @date 2017/10/10
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import os
import unittest
from img_align.test import TrackingExperiment
from img_align.test import TrackingExperimentsSet
import img_align.motion_models
import img_align.object_models
import img_align.cost_functions
import img_align.optimizers

class TestPlanarTracking(unittest.TestCase):

    def testVisualTrackingDataset(self):
        '''
        Test tracking algorithms over Visual Tracking Dataset from paper:

        "Evaluation of Interest Point Detectors and Features Descriptors for Visual Tracking"
        Steffen Gauglitz, Tobias H\"ollerer, Matthew Turk. ICCV 2011.

        The dataset can be found at:

        https://ilab.cs.ucsb.edu/tracking_dataset_ijcv/
        '''

        # BRICKS part of the dataset:
        exp_set = TrackingExperimentsSet()
        exp_path = os.path.join('resources', 'visual_tracking_dataset', 'bricks')
        for exp_file in os.listdir(exp_path):
            if exp_file.endswith('.xml'):
                e = TrackingExperiment(os.path.join(exp_path, exp_file))
                e.load()
                exp_set.add(e)

        exp_set.run()
        results_dir = os.path.join('resources', 'bricks', 'results')
        gt_dir = os.path.join('resources', 'bricks', 'ground_truth')
        evaluation_dir = os.path.join('resources', 'bricks', 'evaluation')
        exp_set.evaluateResults(gt_dir, results_dir, evaluation_dir)


