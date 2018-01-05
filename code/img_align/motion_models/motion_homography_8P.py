
# @brief  Homography 2D motion model (function) for Inverse Compositional tracking
# @author Jose M. Buenaposada
# @date 2017/08/14 (Modified)
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


class MotionHomography8P(MotionModel):
    """
    A class for the 2D homography (8 parameters) motion model.
    The model is:

                 [a   b   c]
        x(k+1) = [d   e   f] * x(k)
                 [g   h   1]

    We use homogeneous coordinates in the model to hand linear equation.
    Where: - x(k) is the position vector in the image plane of the image
             template center in the instant k. x(k) is arranged in a 3
             component vector (homogeneous coordinates)
           - H(a,b,c,d,e,f,g,h) is the matrix which describes a linear change of coordinates

    The parameters are arranged in a 8x1 vector:

                        ( a )
                        ( b )
                        ( c )
                        ( d )
                        ( e )
                        ( f )
                        ( g )
                        ( h )

    Note that when all parameters are a=1, e=1 and the rest 0, the homography matrix is the
    identity matrix.
    """
    def __init__(self):
        super(MotionHomography8P, self).__init__()
        return

    def invMap(self, coords, motion_params):
        """
        Using the motion_params, transform the input coords from image
        coordinates to the corresponding template coordinates.

        The motion params are always from template to current image and with this
        method we have to use the inverse motion model.

        :param coords: current coords in input image reference system
        :param motion_params: current motion params from template to image
        :return: np array with the updated coords.
        """
        H = np.append(motion_params, 1.0).reshape(3, 3)
        invH = np.linalg.pinv(H)
        inv_params = np.copy(np.reshape(invH, (9, 1)))
        inv_params = inv_params[0:8, :]

        return self.map(coords, inv_params)

    def map(self, coords, motion_params):
        """
        This method transform N 2D points given the motion params with the
        given motion model.

        :param coords: current coords in template reference system (Nx2)
        :param motion_params: current motion params from template to image
        :return: np array with the updated cartesian coords (Nx2)
        """

        coords_cols = coords.shape[1]
        coords_rows = coords.shape[0]
        assert(coords_cols == 2)  # We need two dimensional coordinates

        H = np.reshape(np.append(motion_params, 1.0), (3, 3))
        homog_coords = np.ones((coords_rows, 3), dtype=np.float64)
        homog_coords[:, 0:2] = coords
        homog_new_coords = np.dot(homog_coords, H.T)

        # Divide by the third homogeneous coordinates to get the cartesian coordinates.
        third_coord = homog_new_coords[:, 2]
        homog_new_coords = np.copy(homog_new_coords / third_coord[:, np.newaxis])

        return homog_new_coords[:, 0:2]

    def getCompositionParams(self, motion_params1, motion_params2):
        """
        Le f(x, p) = H(p)*(x, 1)^T the motion model function, with motion parameters p, that
        transforms coordinates x. This method computes the motion model params, composition_params,
        such as:
            f(x, composition_params) =  H(motion_params1)*H(motion_params_2)*(x, 1)^T

        :param motion_params1: motion params for f(x, motion_params1).
        :param motion_params2: motion params for second application of f.
        :return: np array with the updated motion parameters.
        """

        H1 = np.reshape(np.append(motion_params1,  1.0), (3, 3))
        H2 = np.reshape(np.append(motion_params2, 1.0), (3, 3))

        Hcomp = np.dot(H1, H2)
        Hcomp = Hcomp / Hcomp[2, 2]  # We make again 1 the coordinate [2, 2] in Hcomp.

        composition_params = np.copy(np.reshape(Hcomp, (9, 1)))
        return composition_params[0:8, :]

    def getCompositionWithInverseParams(self, motion_params1, motion_params2):
        """
        Le f(x, p) = H(p)*(x, 1)^T the motion model function, with motion parameters p, that
        transforms coordinates x. This method computes the motion model params, composition_params,
        such as:
            f(x, composition_params) =  H(motion_params1)*H(motion_params_2)^{-1}*(x, 1)^T

        :param motion_params1: motion params for application of f^{-1}.
        :param motion_params2: motion params for f(x, motion_params1).
        :return: np array with the updated motion parameters.
        """

        H1 = np.reshape(np.append(motion_params1, 1.0), (3, 3))
        H2 = np.reshape(np.append(motion_params2, 1.0), (3, 3))

        Hcomp = np.dot(H1, np.linalg.pinv(H2))
        Hcomp = Hcomp / Hcomp[2, 2]  # We make again 1 the coordinate [2, 2] in Hcomp.

        composition_params = np.copy(np.reshape(Hcomp, (9, 1)))
        return composition_params[0:8, :]

    def computeJacobian(self, coords, motion_params):
        """
        Let f(c, p) the motion model function, with motion parameters p, that
        transforms coordinates c (2 dimensions). This method computes the first derivative
        for the motion model, at given coordinates and with given motion params.

        The function computes the jacobian for N points of dimension 2.

        :param coords: current coords in template reference system (Nx2)
        :param motion_params: current motion params from template to image (8x1)
        :return: np array with the jacobians (2x8xN)
        """
        coords_cols = coords.shape[1]
        coords_rows = coords.shape[0]
        assert(coords_cols == 2) # We need two dimensional coordinates

        H = np.reshape(np.append(motion_params, 1.0), (3, 3))

        # As we start from cartesian coordinates, the 3rd coordinate in
        # the conversion to homogeneous is always 1.
        jacobians_mat = np.zeros((2, 8, coords_rows))
        for i in range(coords_rows):
            # Let p(x_h) = (i/k, j/k) where x_h is a point in homogeneous coordinates.
            # When we have a point in cartesian coordinates k=1.
            # Jp is the derivative of p(x_h) w.r.t. x_h coordinates
            # evaluated for (x, y, 1) homogeneous points. For ith point,
            # x is the coords[i,0] and y is coords[i,1]
            x = coords[i,0]
            y = coords[i,1]
            Jp = np.array([[1., 0., -x], [0., 1., -y]])

            # Jf is the derivative of f(x_h, p) w.r.t. p, evaluated at motion_params:
            Jf = np.array([[x,  y,  1., 0., 0., 0., 0., 0.],
                           [0., 0., 0., x,  y,  1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., x , y]])

            # The Jacobian of f with respect to motion parameters is
            # (d p(x_h) / dx_h) * (d f(x_h, p) / d p)
            jacobians_mat[:,:,i] = np.dot(Jp, Jf)

        return jacobians_mat

    def scaleParams(self, motion_params, scale):
        """
        :param motion_params: current params
        :param scale: scalar value than changes the resolution of the image
        :return: np array with the updated motion params to the new image scale
        """

        H = np.reshape(np.append(motion_params, 1.0), (3, 3))
        S = np.eye(3, 3)
        S[0, 0] = scale
        S[1, 1] = scale
        newH = np.dot(S, H)
        new_params = np.copy(np.reshape(newH, (9, 1)))
        return new_params[0:8, :]

    def getIdentityParams(self):
        """
        :return the motion params that does not change the coordinates in map:
        """
        return np.array([[1.], [0.], [0.],
                         [0.], [1.], [0.],
                         [0.], [0.]])

    def computeParams(self, points_orig, points_dest):
        """
        :param points_orig
        :param points_dest
        :return the motion params that maps points_orig into points_dest:
        """
        H = cv2.getPerspectiveTransform(np.float32(points_orig), np.float32(points_dest))
        H = H / H[2, 2]

        params = np.copy(np.reshape(H, (9, 1)))
        params = params[0:8, :]

        return params

    def getNumParams(self):
        return 8

    def validParams(self, motion_params):
        H = np.reshape(np.append(motion_params, 1.0), (3, 3))

        singular_values = cv2.SVDecomp(H)[0]
        rank = np.sum(singular_values > 1.10-6)

        points = np.array([[0., 0.],
                           [100., 0.],
                           [100., 100.],
                           [0., 100.]])

        transformed_points = self.map(points, motion_params)
        detH = np.linalg.det(H)

        #(detH < 1./10.) or (detH > 10.) or
        return (rank >= 3) and\
               (self.__consistentPoints(points, transformed_points, 0, 1, 2)) and\
               (self.__consistentPoints(points, transformed_points, 1, 2, 3)) and\
               (self.__consistentPoints(points, transformed_points, 0, 2, 3)) and\
               (self.__consistentPoints(points, transformed_points, 0, 1, 3))

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


