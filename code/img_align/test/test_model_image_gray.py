# @brief Object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2017/08
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr


import unittest
import numpy as np
import cv2
import os

from img_align.object_models import ModelImageGray


class TestModelImageGray(unittest.TestCase):

    def setUp(self):
        self.template = cv2.imread(os.path.join('resources', 'book_lowres.jpg'))
        self.image = cv2.imread(os.path.join('resources', 'book_kk_1.jpg'))
        self.model = ModelImageGray(self.template)

    def test_computeTemplateFeatures(self):
        heightDiv2 = round(self.template.shape[0]/2.0)
        widthDiv2 = round(self.template.shape[1]/2.0)
        features = self.model.computeTemplateFeatures()
        ref_coords_system = np.array([widthDiv2, heightDiv2])
        features_img = self.model.computeImageFeatures(self.template, self.model.getReferenceCoords() + ref_coords_system)

        #print 'features.shape = {}'.format(features.shape)
        #print 'features_img.shape = {}'.format(features_img.shape)
        #
        #cv2.imshow('1', np.uint8(np.reshape(features_img, (self.template.shape[0], self.template.shape[1]))))
        #cv2.imshow('2', np.uint8(np.reshape(features, (self.template.shape[0], self.template.shape[1]))))
        #cv2.waitKey()

        self.assertTrue(np.linalg.norm(features - features_img)/255. < 5)

    def test_computeImageFeatures(self):

        # Homography transformation.
        heightDiv2 = round(self.template.shape[0]/2.0)
        widthDiv2 = round(self.template.shape[1]/2.0)
        pts1 = np.array([[-widthDiv2, -heightDiv2],
                          [widthDiv2, -heightDiv2],
                          [widthDiv2, heightDiv2],
                          [-widthDiv2, heightDiv2]], dtype=np.float32)
        pts2 = np.array([[52, 27],
                         [275, 35],
                         [274, 187],
                         [49, 183]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(pts1, pts2)
        H = H / H[2,2]

        coords = self.model.getReferenceCoords()
        homog_coords = np.ones((coords.shape[0], 3), dtype=np.float64)
        homog_coords[:,0:2] = coords
        homog_new_coords = np.dot(homog_coords, H.T)

        # Divide by the third homogeneous coordinates to get the cartesian coordinates.
        third_coord = homog_new_coords[:,2]
        homog_new_coords = np.copy(homog_new_coords / third_coord[:, np.newaxis])

        # Actual computation of image features (resampling from the image).
        features = self.model.computeTemplateFeatures()
        features_img = self.model.computeImageFeatures(self.image, homog_new_coords[:,0:2])

        #cv2.imshow('1', np.uint8(np.reshape(features_img, (self.template.shape[0], self.template.shape[1]))))
        #cv2.imshow('2', np.uint8(np.reshape(features, (self.template.shape[0], self.template.shape[1]))))
        #cv2.waitKey()

        self.assertTrue(np.linalg.norm(features - features_img)/255.0 < 5.0)

    # Fixme: more test needed? what about the image gradients tests?
