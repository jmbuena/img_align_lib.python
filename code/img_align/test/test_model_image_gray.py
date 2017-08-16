import unittest
import numpy as np
import cv2
import os

from img_align.object_models import ModelImageGray

class TestMotionHomography8P(unittest.TestCase):

    def setUp(self):
        self.template = cv2.imread(os.path.join('resources', 'book.jpg'))
        self.model = ModelImageGray(self.template)

    def test_computeTemplateFeatures(self):
        features = self.model.computeTemplateFeatures()
        features_img = self.model.computeImageFeatures(self.template)
        self.assertTrue(np.linalg.norm(features - features_img)[np.newaxis] < 1.0e-10)

    # Fixme: more test needed? what about the image gradients tests?
