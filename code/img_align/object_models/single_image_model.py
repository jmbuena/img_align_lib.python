# @brief Single image object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import cv2
import numpy as np
from img_align.object_models import ObjectModel
from img_align.utils import computeGrayImageGradients


class SingleImageModel(ObjectModel):
    """
    A class that defines single template image (target) model.
    """

    def __init__(self, template_image, equalize=False):
        """

        :param template_image:
        :param equalize:
        """
        super(SingleImageModel, self).__init__()

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

        self.__gradients = computeGrayImageGradients(self.__image)
        self.__gray_levels = np.copy(self.__image.reshape(num_pixels, 1))
        self.__coordinates = self.__computeTemplateCoordinates(self.__image)

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

    def computeTemplateFeaturesGradient(self):
        """
        Computes the grey levels gradient of a template image or any other feature
        in the template model.

        It computes the  \frac{\partial I(\vx)}{\partial \vx} (the gradient).
        The size of the output matrix is Nxk being N the number of pixels and k the
        dimensinality of \vx (the template coordinates vector).

        :return: A np array being Nxk (number of template pixels x dim(\vx) )
        """
        return self.__gradients

    def extractFeaturesFromWarpedImage(self, warped_image):
        """
        Computes the features vector from the warped image.

        Converts the input image to gray levels and then to a column vector (Nx1 with N the
        number of pixels).

        :param warped_image: A np array with the image.
        :return: A np array being Nxk (number of template pixels x 1 )
        """

        # We don't do anything special with the warped image (nor DCT, nor Borders, etc).
        assert (len(warped_image.shape) < 3)
        warpim_cols = warped_image.shape[1]
        warpim_rows = warped_image.shape[0]
        num_pixels = warpim_cols * warpim_rows

        if self.__equalize:
            equalized_image = cv2.equalizeHist(warped_image)
            warped_image_vector = np.copy(equalized_image.reshape(num_pixels, 1))
        else:
            warped_image_vector = np.copy(warped_image.reshape(num_pixels, 1))

        return warped_image_vector

    def computeTemplateFeatures(self, object_params):
        """
        Computes the template gray levels.

        :param object_params: A np array with the motion params
        :return: A vector that is Nx1 (number of template pixels x 1 ).
        """

        gray_levels = np.copy(self.__gray_levels)

        return gray_levels

    def getCtrlPointsIndices(self):
        """
        Returns the index of control points within the reference coordinates vector,

        The coordinates of the top left corner are (-width/2,-height/2),
        and the coordinates of the right bottom corner are  (width/2, height/2). That
        means that the image center pixel take template (reference) coordinats (0,0).

        The control points are the four corners of a rectangle and there are four lines
        that joins the four control points.

        control_points_indices are row indices of reference_coords for the
        control points

        control_points_lines are a list of tuples with the indices of reference points
        (that are control points) joint by the line.

        :return: (control_points_indices, control_points_lines)
        """
        return self.__control_points_indices, self.__control_points_lines

    def getReferenceCoords(self):
        """
        Returns the coordinates of template points (the reference coordinates).

        Returns the coordinates of each pixel from left to right and top to
        bottom (by rows). The coordinates of the top left corner are (-width/2,-height/2),
        and the coordinates of the right bottom corner are  (width/2, height/2). That
        means that the image center pixel take template (reference) coordinats (0,0).

        :return  A np array that is Nx2 (number of template pixels/features x 2)
        """

        return np.copy(self.__coordinates)

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
