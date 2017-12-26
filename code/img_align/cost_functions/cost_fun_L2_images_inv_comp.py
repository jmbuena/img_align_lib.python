
# @brief Optimization problem interface for 'inverse compositional solutions'
# @author Jose M. Buenaposada
# @date 2016/11/12
# @date 2017/08/16 (modified for new CostFunction class)
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import numpy as np
import math
import cv2
from img_align.cost_functions import CostFunL2Images


class CostFunL2ImagesInvComp(CostFunL2Images):

    """
    The interface for the Inverse Compositional cost functions.

    The inverse compositional based problems gets a constant optimization
    problem jacobian J by using a compositional update of motion params.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model, show_debug_info=False):
        """
        :param object_model:
        :param motion_model:
        :return:
        """
        super(CostFunL2ImagesInvComp, self).__init__(object_model, motion_model, show_debug_info)

        self.show_debug_info_jacobians = False
        self.show_debug_info_inv_jacobians = False
        self.__initialized = False
        self.__J = None
        self.__invJ = None

        # NOTE: Here we do not initialize the constant matrices of the Inverse Compositional. They
        # will be only once when needed by calling self.__setupMatrices(). The matrices can not be
        # initialized because Object Models some times are initialized in the first image of a sequence,
        # and with them initialized here we can initialize there (i.e. the image template is cropped from
        # first image) the constant matrices.

    def computeJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """

        if not self.__initialized:
            self.__setupMatrices()

        return self.__J

    def computeJacobianPseudoInverse(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        if not self.__initialized:
            self.__setupMatrices()

        if self.show_debug_info_inv_jacobians:
            for i in range(self.__invJ.shape[0]):
                max_ = np.max(self.__invJ[i,:])
                min_ = np.min(self.__invJ[i,:])
                J_img = self.object_model.convertFeaturesToImage(255*(self.__invJ[i,:]-min_)/(max_-min_))
                cv2.imshow('invJ{}'.format(i), np.uint8(J_img))

        return self.__invJ

    def __setupMatrices(self):
        self.__J = self.__computeConstantJacobian()
        self.__invJ = self.__computeConstantInverseJacobian()

        self.__initialized = True
        return

    def computeValue(self,
                     residual_vector,
                     motion_params,
                     delta_params,
                     image):
        """

        :param residual_vector:
        :param motion_params:
        :param delta_params:
        :param image:
        :return:
        """

        J = self.computeJacobian(motion_params)

        # residual_vector is Nx1 (number of pixels)
        # J is Nxp (number of pixels x number of motion params).
        # delta_params is px1 (number of motion params x 1)
        cost = math.sqrt(np.linalg.norm(residual_vector + np.dot(J, delta_params)))

        return cost

    def computeResiduals(self, image, motion_params):
        """
        :param image: OpenCV np array image.
        :param motion_params: np array 8x1 with the homography 8P params
        :return:
        """

        # We need a 8x1 motion params vector
        assert (motion_params.shape[0] == 8)
        assert (motion_params.shape[1] == 1)

        template_coords = self.object_model.getReferenceCoords()
        image_coords = self.motion_model.map(template_coords, motion_params)
        features_img = self.object_model.computeImageFeatures(image, image_coords)
        # features_template = self.object_model.computeReferenceFeatures(motion_params)
        features_template = self.object_model.computeReferenceFeatures()

        if self.show_debug_info:
            I_warped = self.object_model.convertReferenceFeaturesToImage(features_img)
            I_template = self.object_model.convertReferenceFeaturesToImage(features_template)
            cv2.imshow('features_vector reshaped', I_warped)
            cv2.imshow('template_features_vector reshaped', I_template)

        # The features vector in this case are, Nx2 matrices:
        #   -the first column has the x image gradient
        #   -the second column has the y image gradient
        # assert (features_vector.shape[0] == warped_image.shape[0]*warped_image.shape[1])
        assert (features_img.shape[1] == 1)
        assert (features_template.shape == features_img.shape)

        residuals = np.float64(features_img) - np.float64(features_template)

        return residuals

    def updateMotionParams(self, motion_params, inc_params):
        """
          The updated motion params of an inverse compositional problem is given by
          f(f^{-1}(\vx, delta_params), params) = f(\vx, new_params)

          In the case of a homography motion model is given by:

          H(motion_params) * inverse(H(inc_params + identity_params))

        :param motion_params: old motion parameters
        :param inc_params compositional update paramters vector
        :return: The updated motion params with old ones and the delta in parameters.
        """

        # Compute the compositional update of the motion params, note that the update is done
        # with the parameters increment plus the identity params (this is necessary whenever the
        # motion model has no 0 identity motion parameters).
        identity_params = self.motion_model.getIdentityParams()
        return self.motion_model.getCompositionWithInverseParams(motion_params, inc_params + identity_params)

    def __computeConstantInverseJacobian(self):
        if self.__J is None:
            self.__J = self.__computeConstantJacobian()

        return np.linalg.pinv(self.__J)

    def __computeConstantJacobian(self):

        template_coords = self.object_model.getReferenceCoords()
        J = np.zeros((template_coords.shape[0], self.motion_model.getNumParams()), dtype=np.float64)
        gradients = self.object_model.computeReferenceFeaturesGradient()

        if self.show_debug_info_jacobians:
            max_ = np.max(gradients[:, 0])
            min_ = np.min(gradients[:, 0])
            g_x = self.object_model.convertReferenceFeaturesToImage(255*(np.float32(gradients[:, 0]) - min_)/(max_- min_))
            max_ = np.max(gradients[:, 1])
            min_ = np.min(gradients[:, 1])
            g_y = self.object_model.convertReferenceFeaturesToImage(255*(np.float32(gradients[:, 1]) - min_) / (max_ - min_))
            cv2.imshow('g_x', np.uint8(g_x))
            cv2.imshow('g_y', np.uint8(g_y))

        # The Jacobian of the motion model, in Inverse Compositional, should be instantiated in the
        # motion model identity params.
        identity_params = self.motion_model.getIdentityParams()
        Jf = self.motion_model.computeJacobian(template_coords, identity_params)

        for i in range(Jf.shape[2]):
            J[i, :] = np.dot(gradients[i, :], Jf[:, :, i])

        if self.show_debug_info_jacobians:
            for j in range(J.shape[1]):
                max_ = np.max(J[:, j])
                min_ = np.min(J[:, j])
                J_img = self.object_model.convertReferenceFeaturesToImage(255*(J[:, j]-min_)/(max_-min_))
                cv2.imshow('J{}'.format(j), np.uint8(J_img))

        return J
