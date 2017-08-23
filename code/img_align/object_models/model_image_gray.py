# @brief Single image object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2017/08/16
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import cv2
import numpy as np
import math
from img_align.object_models import ObjectModel
from img_align.utils import computeGrayImageGradients


class ModelImageGray(ObjectModel):
    """
    A class that defines single template image (target) model.
    The template image can be equalized and it is smoothed with
    a Gaussian kernel.
    """

    def __init__(self, template_image, equalize=False):
        """
        :param template_image:
        :param equalize:
        """
        super(ModelImageGray, self).__init__()

        assert (template_image is not None)
        rows = template_image.shape[0]
        cols = template_image.shape[1]
        assert (rows != 0)
        assert (cols != 0)
        num_pixels = rows * cols

        if len(template_image.shape) == 3:
            im_gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)
            self.__image = im_gray
        else:
            self.__image = np.copy(template_image)

        self.__image = cv2.GaussianBlur(self.__image, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)

        self.__equalize = equalize
        if self.__equalize:
            self.__image = cv2.equalizeHist(self.__image)

        self.__gradients = np.float64(computeGrayImageGradients(self.__image))
        self.__gray_levels = np.float64(self.__image.reshape(num_pixels, 1))
        self.__coordinates = self.__computeTemplateCoordinates(self.__image)
        self.__min_coords = np.amin(self.__coordinates, axis=0)
        self.__max_coords = np.amax(self.__coordinates, axis=0)

        self.__control_points_indices = []
        self.__control_points_indices.append(0)
        self.__control_points_indices.append(cols - 1)
        self.__control_points_indices.append(cols * rows - 1)
        self.__control_points_indices.append(cols * (rows - 1))

        self.__control_points_lines = []
        for i in range(4):
            p1 = self.__control_points_indices[i]
            p2 = self.__control_points_indices[(i + 1) % 4]
            self.__control_points_lines.append([p1, p2])

    def computeFeaturesGradient(self):
        """
        Computes the grey levels gradient of a template image or any other feature
        in the template model.

        It computes the  \frac{\partial I(\vx)}{\partial \vx} (the gradient).
        The size of the output matrix is NxK being N the number of pixels and K=2 (cartesian
        coodinates in R2).

        :return: A np array being Nx2 (number of template pixels x 2 )
        """
        return self.__gradients

    def computeImageFeatures(self, image, coords):
        """
        Computes the features vector from the images in the given coords.
        Converts the input image to gray levels and then performs bilinear
        interpolation in the position of the coords on the image. (Nx1 with N the
        number of pixels).

        :param image: A np array with the image.
        :param coords: np array of Nx2, N points being 2-dimensional.
        :return: A np array being Nx1 (number of template pixels x 1 )
        """
        assert(coords.shape[1] == 2)

        # We don't do anything special with the warped image (nor DCT, nor Borders, etc).
        if len(image.shape) == 3:
            im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            im_gray = image

        im_cols = im_gray.shape[1]
        im_rows = im_gray.shape[0]
        num_pixels_img = im_cols*im_rows

        # As coords, in the worst case, it will be between 4 integer coordinates on
        # image. We have to find its gray level using the gray levels of the 4 sorrounding pixels =>
        # bilinear gray level interpolation.
        #
        #
        #    g2 +---------------+ g3
        #       |               |
        #  y    |     * (x,y) <-|---------------- image coordinates to interpolate gray levels on.
        #       |               |
        #    g0 +---------------+ g1
        #               x
        #

        # As template top left corner coords are are (-width/2, -height/2) then, inorder
        # to get (0,0) in that corner, we have to add (width/2, height/2) to all coordinates
        # that are going to sample from an image. self.__min_coords are the coordinates of top
        # left corner of the template.
        floor_x = np.floor(coords[:,0])
        ceil_x = np.ceil(coords[:,0])
        floor_y = np.floor(coords[:,1])
        ceil_y = np.ceil(coords[:,1])

        g0 = np.zeros((coords.shape[0],1))
        g1 = np.zeros((coords.shape[0],1))
        g2 = np.zeros((coords.shape[0],1))
        g3 = np.zeros((coords.shape[0],1))

        l  = (floor_y >= 0) & (floor_y < im_rows) & (floor_x >= 0) & (floor_x < im_cols)
        im_gray_flatten = im_gray.flatten()[:,np.newaxis]
        g0[l,:] = np.float32(im_gray_flatten[np.int32(floor_x[l] + im_cols*floor_y[l])])
        g1[l,:] = np.float32(im_gray_flatten[np.int32(floor_x[l] + im_cols*ceil_y[l])])
        g2[l,:] = np.float32(im_gray_flatten[np.int32(ceil_x[l] + im_cols*ceil_y[l])])
        g3[l,:] = np.float32(im_gray_flatten[np.int32(ceil_x[l] + im_cols*floor_y[l])])

        x_delta = coords[:,0] - floor_x[:]
        x_delta = x_delta[:,np.newaxis]
        y_delta = coords[:,1] - floor_y[:]
        y_delta = y_delta[:,np.newaxis]

        g01 = g0 + (g1 - g0)*x_delta
        g23 = g2 + (g3 - g2)*y_delta

        features = g01 + (g23 - g01)*y_delta

        warped_img = np.uint8(np.reshape(features, (self.__image.shape[0], self.__image.shape[1])))
        warped_img = cv2.GaussianBlur(warped_img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)
        if self.__equalize:
            warped_img = np.float64(cv2.equalizeHist(warped_img))

        num_template_pixels = self.__image.shape[0] * self.__image.shape[1]
        features = np.reshape(warped_img, (num_template_pixels, 1))

        return features

    def computeTemplateFeatures(self, object_params=None):
        """
        Returns the template gray levels as a vector.

        :param object_params: A np array with the motion params
        :return: A vector that is Nx1 (number of template pixels x 1 ).
        """

        gray_levels = np.copy(self.__gray_levels)

        return gray_levels


    def getReferenceCoords(self):
        """
        Returns the coordinates of template points (the reference coordinates).

        Returns the coordinates of each pixel from left to right and top to
        bottom (by rows). The coordinates of the top left corner are (-width/2,-height/2),
        and the coordinates of the right bottom corner are  (width/2, height/2). That
        means that the image center pixel take template (reference) coordinates (0,0).

        :return  A np array that is Nx2 (number of template pixels/features x 2)
        """

        return np.copy(self.__coordinates)


    def getCtrlPointsIndices(self):
        """
        Returns the index of control points within the reference coordinates vector,

        The coordinates of the top left corner are (-width/2,-height/2),
        and the coordinates of the right bottom corner are  (width/2, height/2). That
        means that the image center pixel take template (reference) coordinates (0,0).

        The control points are the four corners of a rectangle and there are four lines
        that joins the four control points.

        control_points_indices are row indices of reference_coords for the
        control points

        control_points_lines are a list of tuples with the indices of reference points
        (that are control points) joint by the line.

        :return: (control_points_indices, control_points_lines)
        """
        return self.__control_points_indices, self.__control_points_lines


    def getNumOfReferenceCoords(self):
        return self.__image.shape[0] * self.__image.shape[1]


    def __computeTemplateCoordinates(self, gray_image):

        assert (len(gray_image.shape) < 3)

        rows = gray_image.shape[0]
        cols = gray_image.shape[1]
        width_div_2 = round(cols / 2.0)
        height_div_2 = round(rows / 2.0)

        coords_list = [[j - width_div_2, i - height_div_2] for i in range(rows) for j in range(cols)]
        coordinates = np.array(coords_list)

        return coordinates
