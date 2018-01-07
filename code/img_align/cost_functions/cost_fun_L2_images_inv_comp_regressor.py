
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
import matplotlib.pyplot as plt
from img_align.cost_functions import CostFunL2ImagesInvComp


class CostFunL2ImagesInvCompRegressor(CostFunL2ImagesInvComp):

    """
    The interface for the Inverse Compositional with regression cost functions.

    The inverse compositional based problems gets a constant optimization
    problem jacobian J by using a compositional update of motion params.
    In this case, the difference with the just 'Inverse Compositional' is that the
    Jacobian pseudoinverse is computed by linear least squares regression. See
    papers:

         Hyperplane approximation for template matching
         F. Jurie, M. Dhome
         PAMI 24(7) 2002.

         and

         Online Learning of Linear Predictors for Real-Time Tracking
         S. Holzer, M. Pollefeys, S. Illic, D. Tan, N. Navab
         ECCV 2012.


    On the other hand, the Jurie's paper seems to make an additive update of the parameters,
    but in order to have a constant Jacobian you need to use the Inverse Compositional approach
    (i.e. composition of motion parameters and computing the motion increment over the
    template).

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model, num_samples=20000, show_debug_info=False, shake_corners=False):
        """
        :param object_model: Object model to use in tracking
        :param motion_model: Motion model to use in tracking.
        :param num_samples: Number os samples (motion params, image features differences) to generate
        :param shake_corners: move randomly the four corners of the template to generate samples
        :return:
        """
        super(CostFunL2ImagesInvCompRegressor, self).__init__(object_model, motion_model, show_debug_info)

        self.show_debug_info_jacobians = show_debug_info
        self.show_debug_info_inv_jacobians = show_debug_info
        self.show_debug_info_regressor = False
        self.num_samples = num_samples
        self.shake_corners = shake_corners
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

        if self.show_debug_info_jacobians:
            for j in range(self.__J.shape[1]):
                max_ = np.max(self.__J[:, j])
                min_ = np.min(self.__J[:, j])
                J_img = self.object_model.convertReferenceFeaturesToImage(255*(self.__J[:, j]-min_)/(max_-min_))
                cv2.imshow('J{}'.format(j), np.uint8(J_img))

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

    def __generateDeltaParamsCorners(self, template_reference_coords, template_coords,
                                     template_image, num_samples, object_model, motion_model):
        """
        Generates the delta motion params for the template image. Note that there are two
        types of template coords:

          - The reference template coords. This coordinates are for a rectified template image
            with the origin of 2D coordinates in the center of the template (the top left corner is
            [-width/2, -heigh/2].
          - The template coords. This coordinates are the transformed reference coordinates over the
            image with the template embedded (e.g. an image of a book on a table).

        The increment in motion parameters are computed over the reference template coords. This is the key
        for this method to work (it is the same point at which we compute the motion model jacobian
        in the Inverse Compositional implementation we have).

        :param template_reference_coords:
        :param template_coords:
        :param template_image:
        :param num_samples:
        :param motion_model:
        :return:
        """

        ctrl_indices, ctrl_lines = object_model.getCtrlPointsIndices()

        Y = np.zeros((motion_model.getNumParams(), num_samples))

        orig_ref_pts = np.zeros((len(ctrl_indices), 2))
        for i in range(orig_ref_pts.shape[0]):
            orig_ref_pts[i, 0] = template_reference_coords[ctrl_indices[i], 0]
            orig_ref_pts[i, 1] = template_reference_coords[ctrl_indices[i], 1]

        orig_pts = np.zeros((len(ctrl_indices), 2))
        for i in range(orig_pts.shape[0]):
            orig_pts[i, 0] = template_coords[ctrl_indices[i], 0]
            orig_pts[i, 1] = template_coords[ctrl_indices[i], 1]

        params_ref2orig = motion_model.computeParams(orig_ref_pts, orig_pts)

        show_transformations = False

        identity_params = motion_model.getIdentityParams()
        for i in range(num_samples):
            # Shake randomly the four corners (a little bit)
            delta_pts = np.random.uniform(low=-3.0, high=3.0, size=(orig_pts.shape[0], 2))
            dst_pts = orig_ref_pts + delta_pts

            # Compute motion params from origin and modified points
            orig_ref2dst_params = motion_model.computeParams(orig_ref_pts, dst_pts)
            homography = np.reshape(np.append(orig_ref2dst_params, 1.0), (3, 3))
            new_params = np.copy(np.reshape(homography, (9, 1)))
            new_params = new_params[0:8, :]

            # Subtract identity params in order to keep all with zeros params vector as the identity
            # transformation (needed in inverse compositional).
            delta_params = new_params - identity_params
            Y[:, i] = delta_params.T

            if show_transformations:
                template_image_copy = template_image.copy()
                for j in range(orig_pts.shape[0]):
                    cv2.line(template_image_copy,
                             (int(orig_pts[j, 0]), int(orig_pts[j, 1])),
                             (int(orig_pts[(j + 1) % 4, 0]), int(orig_pts[(j + 1) % 4, 1])),
                             color=(0, 0, 255),  # red color
                             thickness=2)

                for j in range(dst_pts.shape[0]):
                    cv2.line(template_image_copy,
                             (int(dst_pts[j, 0]), int(dst_pts[j, 1])),
                             (int(dst_pts[(j + 1) % 4, 0]), int(dst_pts[(j + 1) % 4, 1])),
                             color=(255, 255, 255),  # white color
                             thickness=1)

                cv2.imshow("template image", template_image_copy)
                cv2.waitKey()

        return Y, params_ref2orig

    def __generateDeltaParamsRandomParams(self, template_reference_coords, template_coords,
                                         template_image, num_samples, object_model, motion_model):
        """
        Generates the delta motion params for the template image. Note that there are two
        types of template coords:

          - The reference template coords. This coordinates are for a rectified template image
            with the origin of 2D coordinates in the center of the template (the top left corner is
            [-width/2, -heigh/2].
          - The template coords. This coordinates are the transformed reference coordinates over the
            image with the template embedded (e.g. an image of a book on a table).

        The increment in motion parameters are computed over the reference template coords. This is the key
        for this method to work (it is the same point at which we compute the motion model jacobian
        in the Inverse Compositional implementation we have).

        :param template_reference_coords:
        :param template_coords:
        :param template_image:
        :param num_samples:
        :param motion_model:
        :return:
        """

        ctrl_indices, ctrl_lines = object_model.getCtrlPointsIndices()

        orig_ref_pts = np.zeros((len(ctrl_indices), 2))
        for i in range(orig_ref_pts.shape[0]):
            orig_ref_pts[i, 0] = template_reference_coords[ctrl_indices[i], 0]
            orig_ref_pts[i, 1] = template_reference_coords[ctrl_indices[i], 1]

        orig_pts = np.zeros((len(ctrl_indices), 2))
        for i in range(orig_pts.shape[0]):
            orig_pts[i, 0] = template_coords[ctrl_indices[i], 0]
            orig_pts[i, 1] = template_coords[ctrl_indices[i], 1]

        params_ref2orig = motion_model.computeParams(orig_ref_pts, orig_pts)
        Y = motion_model.generateRandomParamsIncrements(num_samples, n_sigmas=1)

        return Y, params_ref2orig

    def __computeConstantInverseJacobian(self):

        template_image, template_coords = self.object_model.getTemplateImageAndCoords()
        template_reference_coords = self.object_model.getReferenceCoords()

        # Matrix with the motion params of the generated samples by columns: p x num_samples (p is motion parameters)
        # Y, ref2img_params = self.__generateDeltaParams(template_reference_coords, template_coords, template_image,
        #                                                self.num_samples, self.object_model, self.motion_model)
        # Y, homography_ref2img = self.__generateDeltaParams(template_reference_coords, template_coords, template_image,
        #                                                self.num_samples, self.object_model, self.motion_model)
        # ref2img_params = np.copy(np.reshape(homography_ref2img, (9, 1)))
        # ref2img_params = ref2img_params[0:8, :]
        if self.shake_corners:
            Y, ref2img_params = self.__generateDeltaParamsCorners(template_reference_coords, template_coords, template_image,
                                                                  self.num_samples, self.object_model, self.motion_model)
        else:
            Y, ref2img_params = self.__generateDeltaParamsRandomParams(template_reference_coords, template_coords, template_image,
                                                                       self.num_samples, self.object_model, self.motion_model)

        # H is N x num_samples (number of pixels x number_samples generated)
        H = np.zeros((template_coords.shape[0], self.num_samples), dtype=np.float64)

        # generate samples around the identity parameters.
        features_template = self.object_model.computeReferenceFeatures()

        identity_params = self.motion_model.getIdentityParams()
        for i in range(self.num_samples):
            # we are going to use a uniform sampling scheme for the motion parameters.
            delta_params = np.reshape(Y[:, i], (self.motion_model.getNumParams(), 1))

            delta_params_transform = identity_params + delta_params
            params = self.motion_model.getCompositionParams(delta_params_transform, ref2img_params)

            coords = self.motion_model.map(template_reference_coords, params)

            features_img = self.object_model.computeImageFeatures(template_image, coords)
            delta_gray_i = (np.float64(features_img) - np.float64(features_template))
            H[:, i] = delta_gray_i.T

            if i % 1000 == 0:
                print '{}'.format(i)

            if self.show_debug_info_regressor:
                template_image_copy = template_image.copy()
                for j in range(coords.shape[0]):
                    cv2.circle(template_image_copy,
                               (int(coords[j, 0]), int(coords[j, 1])), 1, (0, 0., 255.),  # red color
                               -1)  # filled
                cv2.imshow("Points", template_image_copy)

                features_img_show = self.object_model.convertReferenceFeaturesToImage(features_img)
                warped_image_show = self.object_model.convertReferenceFeaturesToImage(features_img_show)
                cv2.imshow("Warped Image", warped_image_show)

                features_template_show = self.object_model.convertReferenceFeaturesToImage(features_template)
                template_image_show = self.object_model.convertReferenceFeaturesToImage(np.uint8(features_template_show))
                cv2.imshow("Template Image", template_image_show)

                max_ = np.max(delta_gray_i)
                min_ = np.min(delta_gray_i)
                diff_image_show = self.object_model.convertReferenceFeaturesToImage(255*(np.float64(delta_gray_i)-min_)/(max_-min_))
                cv2.imshow("Diff Image", diff_image_show)

                cv2.waitKey()

        # -----------------------------------------------------------------------
        # Direct application of Jurie-Dhome approach: very slow training.
        # -----------------------------------------------------------------------
        # Note: H2 can be rank deficient as it is num_pixels x num_pixels matrix and we are using self.num_samples
        #       columns in delta_gray (delta_gray is num_pixels x self..num_samples).
        #H2 = np.dot(H, H.T)
        #invH2 = np.linalg.inv(H2)
        #pinvH = np.dot(H.T, invH2)
        #A = np.dot(Y, pinvH)
        #return A

        # -----------------------------------------------------------------------
        # Application of Holzer et al. method which results in faster training:
        #   "Online Learning of Linear Predictors for Real-Time Tracking"
        #  S. Holzer, M. Pollefeys, S. Illic, D. Tan, N. Navab
        #  ECCV 2012.
        # -----------------------------------------------------------------------
        pinvY = np.dot(Y.T, np.linalg.inv(np.dot(Y, Y.T)))
        B = np.dot(H, pinvY)
        A = np.linalg.pinv(B)
        return A

    def __computeConstantJacobian(self):

        if self.__invJ is None:
            self.__invJ = self.__computeConstantInverseJacobian()

        return np.linalg.pinv(self.__invJ)
