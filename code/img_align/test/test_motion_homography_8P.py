import unittest
import numpy as np
from math import *

from img_align.motion_models import MotionHomography8P

class TestMotionHomography8P(unittest.TestCase):

    def setUp(self):
        self.motion = MotionHomography8P()

    def test_map_identity(self):
        # We need 2D coordinates in a column vector.
        p = np.array([3, 4])[np.newaxis]
        # Params of the homography are given in row major order.
        params = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        p_map = self.motion.map(p, params)

        # Test passed if the p_map is equal to (3,4)
        self.assertTrue(np.linalg.norm(p_map - np.array([3, 4])[np.newaxis]) < 1.0e-10)

    def test_map_rotation(self):
        # We need 2D coordinates in a column vector.
        p = np.array([1, 0])[np.newaxis]
        # Params of the homography are given in row major order.
        params = np.array([cos(pi/4), -sin(pi/4), 0, sin(pi/4), cos(pi/4), 0, 0, 0])
        p_map = self.motion.map(p, params)

        self.assertTrue(np.linalg.norm(p_map - np.array([cos(pi/4), cos(pi/4)])[np.newaxis]) < 1.0e-10)

    def test_map_rotation_3_points(self):
        # We need 2D coordinates in a column vector.
        p = np.array([[1, 0], [0, 1], [-1, 0]])
        # Params of the homography are given in row major order.
        params = np.array([cos(pi/4), -sin(pi/4), 0, sin(pi/4), cos(pi/4), 0, 0, 0])
        p_map = self.motion.map(p, params)

        p_map_gt = np.array([[cos(pi/4), sin(pi/4)], [-sin(pi/4), cos(pi/4)], [-cos(pi/4), -sin(pi/4)]])
        self.assertTrue(np.linalg.norm(p_map - p_map_gt) < 1.0e-10)

    def test_invMap(self):
        # We need 2D coordinates in a column vector.
        p = np.array([[cos(pi/4), sin(pi/4)], [-sin(pi/4), cos(pi/4)], [-cos(pi/4), -sin(pi/4)]])
        # Params of the homography are given in row major order.
        params = np.array([cos(pi/4), -sin(pi/4), 0, sin(pi/4), cos(pi/4), 0, 0, 0])
        p_map = self.motion.invMap(p, params)

        p_map_gt = np.array([[1, 0], [0, 1], [-1, 0]])
        self.assertTrue(np.linalg.norm(p_map - p_map_gt) < 1.0e-10)

    def test_getCompositionParams_3_points(self):

        # Params of the homography are given in row major order.
        params1 = np.array([cos(pi/4), -sin(pi/4), 0, sin(pi/4), cos(pi/4), 0, 0, 0]) # rotation
        params2 = np.array([1, 0, -100, 0, 1, 200, 0, 0]) # traslation
        comp_params = self.motion.getCompositionParams(params1, params2)

        comp_params_gt = np.array([cos(pi/4), -sin(pi/4), -100, sin(pi/4), cos(pi/4), 200, 0, 0])
        self.assertTrue(np.linalg.norm(comp_params - comp_params_gt) < 1.0e-10)

    def test_getCompositionParams2_3_points(self):

        # Params of the homography are given in row major order.
        params1 = np.array([cos(pi/4), -sin(pi/4), 0, sin(pi/4), cos(pi/4), 0, 0, 0]) # rotation +pi/4
        params2 = np.array([cos(pi/4), sin(pi/4), 0, -sin(pi/4), cos(pi/4), 0, 0, 0]) # rotation -pi/4
        comp_params = self.motion.getCompositionParams(params1, params2)

        comp_params_gt = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(np.linalg.norm(comp_params - comp_params_gt) < 1.0e-10)

    def test_scaleParams(self):
        params = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        params_scaled = self.motion.scaleParams(params, 3.0)
        params_scaled_gt = np.array([3, 0, 0, 0, 3, 0, 0, 0])

        self.assertTrue(np.linalg.norm(params_scaled - params_scaled_gt) < 1.0e-10)

    def test_getNumParams(self):
        self.assertTrue(self.motion.getNumParams() == 8)

    def test_validParams(self):
        return