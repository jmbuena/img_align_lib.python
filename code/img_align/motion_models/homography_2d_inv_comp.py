
# @brief  Homography 2D motion model (function) for Inverse Compositional tracking
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import numpy as np
import cv2
from img_align.motion_models import MotionModel


class Homography2DInvComp(MotionModel):
    """
    A class for the 2D homography (8 parameters) motion model.
    The model is:

                 [a+1 b   c]
        x(k+1) = [d   e+1 f] * x(k)
                 [g   h   i+1]

    We use homogeneus coordinates in the model to hand linear equation.
    Where: - x(k) is the posision vector in the image plane of the image
             template center in the instant k. x(k) is arranged in a 3
             component vector (homogeneus coordinates)
           - H(a,b,c,d,e,f,g,h,i) is the matrix which describes a
             linear change of coordinates

    The parameters are arranged in a 9x1 vector:

                        ( a )
                        ( d )
                        ( g )
                        ( b )
                        ( e )
                        ( h )
                        ( c )
                        ( f )
                        ( i )

    Note that when all parameters are 0, the homography matrix is the
    identity matrix.
    """
    def __init__(self):
        super(Homography2DInvComp, self).__init__()
        return

    def scaleInputImageResolution(self, motion_params, scale):
        """
        :param motion_params: current params
        :param scale: scalar value than changes the resolution of the image
        :return: np array with the updated motion params to the new image scale
        """

        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to add np.eye(3,3)
        H = np.reshape(motion_params, (3,3)).T + np.eye(3,3)
        S = np.eye(3,3)
        S[0,0] = scale
        S[1,1] = scale
        newH = np.dot(S, H)
        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to subtract np.eye(3,3) to get the
        # parameters back from the homography matrix.
        newH = newH - np.eye(3,3)
        new_params = np.copy(np.reshape(newH.T, (9,1)))

        return new_params

    def transformCoordsToTemplate(self, coords, motion_params):
        """
        Using the motion_params, transform the input coords from image
        coordinates to the corresponding template coordinates.

        The motion params are always from template to current image and with this
        method we have to use the inverse motion model.

        :param coords: current coords in input image reference system
        :param motion_params: current motion params from template to image
        :return: np array with the updated coords.
        """
        H = motion_params.reshape(3,3).T + np.eye(3,3)
        invH = H.inv() - np.eye(3,3)
        inv_params = np.copy(np.reshape(invH.T, (9,1)))

        return self.transformCoordsToImage(coords, inv_params)

    def transformCoordsToImage(self, coords, motion_params):
        """

        :param coords: current coords in template reference system
        :param motion_params: current motion params from template to image
        :return: np array with the updated cartesian coords (Nx2)
        """
        coords_cols = coords.shape[1]
        coords_rows = coords.shape[0]
        assert(coords_cols == 2); # We need two dimensional coordinates

        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to add np.eye(3,3)
        H = np.reshape(motion_params, (3,3)).T + np.eye(3,3)
        homog_coords = np.ones((coords_rows, 3), dtype=np.float64)
        homog_coords[:,0:2] = coords

        homog_new_coords = np.dot(homog_coords, H.T)

        # Divide by the third homogeneous coordinates to get the cartersian coordinates.
        third_coord = homog_new_coords[:,2]
        homog_new_coords = np.copy(homog_new_coords / third_coord[:, np.newaxis])

        return homog_new_coords

    def warpImage(self, image, motion_params, template_coords):
        """

        :param image:
        :param motion_params:
        :param template_coords:
        :return:
        """
        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to add np.eye(3,3)
        H = np.reshape(motion_params, (3,3)).T + np.eye(3,3)

        # Find min_x, min_y as well as max_x,max_y in template coords.
        max_val = np.amax(template_coords, axis=0)
        min_val = np.amin(template_coords, axis=0)

        TR = np.array([[1., 0., min_val[0]], # min_x
                       [0., 1., min_val[1]], # min_y
                       [0., 0., 1.]])

        # TR is necessary because the Warpers do warping taking
        # he pixel (0,0) as the left and top most pixel of the template.
        # So, we have move the (0,0) to the center of the Template.
        M = np.dot(H, TR)

        warped_img_rows = np.int(max_val[1] - min_val[1] + 1)
        warped_img_cols = np.int(max_val[0] - min_val[0] + 1)
        warped_image = cv2.warpPerspective(image, M,
                                           (warped_img_cols, warped_img_rows),
                                           flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)

        return warped_image

    def getNumParams(self):
        return 9

    def invalidParams(self, motion_params):
        # The Inverse compositional homography model takes 1 + parameters in the
        # diagonal, therefore we need to add np.eye(3,3)
        H = np.reshape(motion_params, (3,3)).T + np.eye(3,3)

        singular_values = cv2.SVDecomp(H)[0]
        rank = np.sum[singular_values > 1.10-6]

        points = np.array([[0., 0.],
                           [100., 0.],
                           [100., 100.],
                           [0., 100.]])

        transformed_points = np.copy(points)
        transformed_points = self.transformCoordsToImage(points, motion_params)
        detH = np.linalg.det(H)

        #(detH < 1./10.) or (detH > 10.) or
        return (rank<3) or\
               (not self.__consistentPoints(points, transformed_points, 0, 1, 2)) or\
               (not self.__consistentPoints(points, transformed_points, 1, 2, 3)) or\
               (not self.__consistentPoints(points, transformed_points, 0, 2, 3)) or\
               (not self.__consistentPoints(points, transformed_points, 0, 1, 3))

    def __consistentPoints(self, points, transformed_points, t1, t2, t3):
        """
         This method checks weather a quadrilateral in the template reference system
         (points) is transformed by H in a consistent quadrilateral. A consistent
         quadrilateral is one that keeps the order of the points in a circular fashion
         (i.e. is not transformed in a bow tie shape).

         See paper:

         Speeding-up homography estimation in mobile devices.
         Pablo M'arquez-Neila, Javier L'opez-Alberca, Jos'e M. Buenaposada, Luis Baumela.
         Journal of Real-Time Image Processing, 11(1): 141-154 (2016)

        :param points:
        :param transformed_points:
        :param t1:
        :param t2:
        :param t3:
        :return:
        """
        mA = np.array([[points[t1, 0], points[t1, 1]],
                       [points[t2, 0], points[t2, 1]],
                       [points[t3, 0], points[t3, 1]]])

        mB = np.array([[transformed_points[t1, 0], transformed_points[t1, 1]],
                       [transformed_points[t2, 0], transformed_points[t2, 1]],
                       [transformed_points[t3, 0], transformed_points[t3, 1]]])

        detA = np.linalg.det(mA)
        detB = np.linalg.det(mB)

        return detA * detB >= 0.
