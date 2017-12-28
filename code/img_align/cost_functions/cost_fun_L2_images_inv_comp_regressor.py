
# @brief Optimization problem interface for 'inverse compositional solutions' with learned Jacobian pseudoinverse
# @author Jose M. Buenaposada
# @date 2017/12/21
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
from img_align.cost_functions import CostFunL2ImagesInvComp


class CostFunL2ImagesInvCompRegressor(CostFunL2ImagesInvComp):

    """
    The interface for the Inverse Compositional with regression cost functions.

    The inverse compositional based problems gets a constant optimization
    problem jacobian J by using a compositional update of motion params.
    In this case, the difference with the just 'Inverse Compositional' is that the
    Jacobian pseudoinverse is computed by linear least squares regression. See
    paper:

         Hyperplane approximation for template matching
         F. Jurie, M. Dhome
         PAMI 24(7) 2002.

    On the other hand, the paper seems to make an additive update of the parameters, but
    in order to have a constant Jacobian you need to use the Inverse Compositional approach
    (i.e. composition of motion parameters and computing the motion increment over the
    template).

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model, num_samples=1000, show_debug_info=False):
        #low = None, high = None,
        """
        :param object_model: Object model to use in tracking
        :param motion_model: Motion model to use in tracking.
        :param num_samples: Number os samples (motion params, image features differences) to generate
        :return:
        """
        super(CostFunL2ImagesInvCompRegressor, self).__init__(object_model, motion_model, show_debug_info)

        self.show_debug_info_jacobians = True
        self.show_debug_info_inv_jacobians = True
        self.num_samples = num_samples
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

        # if self.show_debug_info_jacobians:
        #     for j in range(self.__J.shape[1]):
        #         max_ = np.max(self.__J[:, j])
        #         min_ = np.min(self.__J[:, j])
        #         J_img = self.object_model.convertReferenceFeaturesToImage(255*(self.__J[:, j]-min_)/(max_-min_))
        #         cv2.imshow('J{}'.format(j), np.uint8(J_img))

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
                max_ = np.max(self.__invJ[i, :])
                min_ = np.min(self.__invJ[i, :])
                J_img = self.object_model.convertReferenceFeaturesToImage(255*(self.__invJ[i,:]-min_)/(max_-min_))
                cv2.imshow('invJ{}'.format(i), np.uint8(J_img))

        return self.__invJ

    def __setupMatrices(self):
        # In this case we compute by regression the inverse jacobian instead of the
        self.__invJ = self.__computeConstantInverseJacobian()
        self.__J = self.__computeConstantJacobian()

        self.__initialized = True
        return

    def __computeConstantInverseJacobian(self):

        template_image, template_coords = self.object_model.getTemplateImageAndCoords()
        #template_coords = self.object_model.getReferenceCoords()
        invJ = np.zeros((self.motion_model.getNumParams(), template_coords.shape[0]), dtype=np.float64)

        # Matrix with the motion params of the generated samples by columns: p x num_samples (p is motion parameters)
        delta_motion_params = np.random.uniform(low=-0.0000001, high=0.0000001,
                                                size=(self.motion_model.getNumParams(), self.num_samples))

        # delta_gray is N x num_samples (number of pixels x number of samples generated)
        delta_gray = np.zeros((template_coords.shape[0], self.num_samples), dtype=np.float64)

        # generate samples around the identity parameters.
        features_template = self.object_model.computeReferenceFeatures()

        identity_params = self.motion_model.getIdentityParams()
        for i in range(self.num_samples):
            # we are going to use a uniform sampling scheme for the motion parameters.
            #motion_params = delta_motion_params[:, i]
            delta_params = np.reshape(delta_motion_params[:, i], (self.motion_model.getNumParams(), 1))
            motion_params = delta_params
            coords = self.motion_model.map(template_coords, motion_params)
            features_img = self.object_model.computeImageFeatures(template_image, coords)
            delta_gray_i = np.float64(features_img - features_template)
            delta_gray[:, i] = delta_gray_i.T
            delta_motion_params[:, i] = np.reshape(identity_params, (self.motion_model.getNumParams(), 1)) + delta_params

            if self.show_debug_info_jacobians:
                # max_ = np.max(features_img)
                # min_ = np.min(features_img)
                #features_img_show = self.object_model.convertReferenceFeaturesToImage(255*(np.float64(features_img)-min_)/(max_-min_))
                features_img_show = self.object_model.convertReferenceFeaturesToImage(features_img)
                warped_image_show = self.object_model.convertReferenceFeaturesToImage(features_img_show)
                cv2.imshow("Warped Image", warped_image_show)

                #max_ = np.max(features_template)
                #min_ = np.min(features_template)
                #features_template_show = self.object_model.convertReferenceFeaturesToImage(255*(np.float64(features_template)-min_)/(max_-min_))
                features_template_show = self.object_model.convertReferenceFeaturesToImage(features_template)
                template_image_show = self.object_model.convertReferenceFeaturesToImage(np.uint8(features_template_show))
                cv2.imshow("Template Image", warped_image_show)

                max_ = np.max(delta_gray_i)
                min_ = np.min(delta_gray_i)
                diff_image_show = self.object_model.convertReferenceFeaturesToImage(255*(np.float64(delta_gray_i)-min_)/(max_-min_))
                cv2.imshow("Diff Image", diff_image_show)

        return np.dot(delta_motion_params, np.linalg.pinv(delta_gray))

    def __computeConstantJacobian(self):

        if self.__invJ is None:
            self.__invJ = self.__computeConstantInverseJacobian()

        return np.linalg.pinv(self.__invJ)
