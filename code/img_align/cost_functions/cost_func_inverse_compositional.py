
# @brief Optimization problem interface for 'inverse compositional solutions'
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc
import numpy as np
from img_align.cost_functions import CostFunction


class CostFuncL2ImagesInvComp(CostFunction):

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
        super(CostFuncL2ImagesInvComp, self).__init__(object_model, motion_model, show_debug_info)

        self.__initialized = False
        self.__J = None
        self.__invJ = None
        self.__setupMatrices()

        return

    def computeJacobian(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """

        if not self.__initialized:
            self.__setupMatrices()

        return self.__J

    def computeJacobianPseudoinverse(self, motion_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        if not self.__initialized:
            self.__setupMatrices()

        return self.__invJ


    def __setupMatrices(self):
        self.__J = self.computeConstantJacobian()
        self.__invJ = np.linalg.pinv(self.__J)

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
        # redidual_vector is Nx1 (number of pixels)
        # J is Nx8 (number of pixels x number of motion params).
        # delta_params is 8x1 (number of motion params x 1)
        cost = np.linalg.norm(residual_vector + np.dot(J, delta_params))

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
        warped_image = self.motion_model.warpImage(image, motion_params, template_coords)

        if self.show_debug_info:
            print "template_coords=", template_coords
            print "warped_image.shape=", warped_image.shape
            cv2.imshow('Warped Image original', warped_image)
            cv2.waitKey()

        if len(warped_image.shape) == 3:
            warped_image_gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
        else:
            warped_image_gray = np.copy(warped_image)

        features_vector = self.object_model.extractImageFeatures(warped_image_gray)
        template_features_vector = self.object_model.computeFeatures(motion_params)

        if self.show_debug_info:
            I_warped = np.uint8(np.reshape(features_vector, warped_image.shape))
            I_template = np.uint8(np.reshape(template_features_vector, warped_image.shape))
            cv2.imshow('features_vector reshaped', I_warped)
            cv2.imshow('template_features_vector reshaped', I_template)

        # The features vector in this case are, Nx2 matrices:
        #   -the first column has the x image gradient
        #   -the second column has the y image gradient
        assert (features_vector.shape[0] == warped_image.shape[0]*warped_image.shape[1])
        assert (features_vector.shape[1] == 1)
        assert (template_features_vector.shape == features_vector.shape)

        residuals = np.float64(features_vector) - np.float64(template_features_vector)

        if self.show_debug_info:
            I_residuals = np.reshape(residuals, warped_image.shape)
            max_residual = np.max(I_residuals)
            min_residual = np.min(I_residuals)
            I_residual = np.uint8(255*(I_residuals - min_residual) / (max_residual - min_residual))
            cv2.imshow('residual reshaped', I_residuals)
            cv2.waitKey()

        return residuals

    def updateMotionParams(self, motion_params, inc_params):
        """
          The updated motion params of an inverse compositional problem is given by
          f(f^{-1}(\vx, delta_params), params) = f(\vx, new_params)

          In the case of a homography motion model is given by:

          H(motion_params) * inverse(H(inc_params))

        :param motion_params: old motion parameters
        :param inc_params compositional update paramters vector
        :return: The updated motion params with old ones and the delta in parameters.
        """

        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to add np.eye(3,3)
        H = np.reshape(motion_params, (3,3)).T + np.eye(3,3)
        dH = np.reshape(inc_params, (3,3)).T + np.eye(3,3)

        newH = np.dot(H, np.linalg.pinv(dH))

        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to subtract np.eye(3,3) to get the
        # parameters back from the homography matrix.
        newH = newH - np.eye(3, 3)
        new_params = np.copy(np.reshape(newH.T, (9, 1)))

        return new_params


    @abc.abstractmethod
    def computeConstantJacobian(self):
        template_coords = self.object_model.getReferenceCoords()
        J = np.zeros((template_coords.shape[0], self.motion_model.getNumParams()), dtype=np.float64)
        zero_params = np.zeros(self.motion_model.getNumParams())
        gradients = self.object_model.computeFeaturesGradient()

        Jf = self.motion_model.computeJacobian(template_coords, self.motion_model.getIdentityParams())

        for i in range(Jf.shape[2]):
           J[i,:] = np.dot(gradients[i,:], Jf[:,:,i])

        return J