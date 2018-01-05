
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
from img_align.object_models import ObjectModel
from img_align.utils import computeGrayImageGradients


class ModelImageGray(ObjectModel):
    """
    A class that defines single template image (target) model.
    The template image can be equalized and it is smoothed with
    a Gaussian kernel.
    """

    def __init__(self,
                 template_image=None, template_image_coords=None,
                 template_image_shape=None, equalize=False):
        """
        This constructor can take a template_image that can be in the form of:
            1) A rectified image. For example, the cover of a book warped
               to have the sides parallel to the image axes and extended to
               the image boundaries.

            2) A whole image where the template is embedded. For example,
               a book cover captured without any restrictions.

        In any of the cases, the template_image param is the image itself. If the
        template_image is None in the constructor then the template image is
        set to the gray levels of the image and coordinates of the first call to
        computeImageFeatures.

        The template_image_coords are the coordinates of the template
        pixels in the template_image. If template_image_coords is None, then the
        constructor assumes that the template_image is a rectified one and the
        template_image_coords are the same as the rectified image ones.

        The template_image_shape defines the size of the rectified template image
        to use in tracking. I will be extracted from the template_image in any case.

        :param template_image: The template image itself
        :param template_image_coords: Nx2 numpy array with 2D coords of the pixels in the template image
        :param template_image_shape: (height, width) tuple with the image dimensions in pixels
        :param equalize: Bool value for equalize or not the template image.
        """
        super(ModelImageGray, self).__init__()
        self.__image = None
        self.__image_rectified = None
        self.__template_image_shape = None
        self.__equalize = equalize
        self.__gradients = None
        self.__gray_levels = None
        self.__coordinates_original = None
        self.__coordinates_rectified = None

        self.__control_points_indices = []
        self.__control_points_lines = []

        if template_image_shape is not None:
            self.__changeRectifiedTemplateSize(template_image_shape)

        if template_image is not None:
            self.setTemplateImage(template_image, template_image_coords)

    def __changeRectifiedTemplateSize(self, template_image_shape):

        rows = template_image_shape[0]
        cols = template_image_shape[1]
        assert (rows != 0)
        assert (cols != 0)
        self.__template_image_shape = template_image_shape

        self.__coordinates_rectified = self.__computeRectifiedTemplateCoordinates(template_image_shape)
        if self.__coordinates_original is None:
            self.__coordinates_original = np.copy(self.__coordinates_rectified)

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

    def setTemplateImage(self, template_image, template_image_coords):

        if template_image is None:
            return

        self.__image = template_image

        template_image_rectified = template_image
        if template_image_coords is not None:
            # The template_image is not rectified image
            # We use the template_image_coords to re-sample the image where the template is embedded
            template_image_rectified = self.__resampleImage(template_image,
                                                            template_image_coords,
                                                            self.__template_image_shape)
        self.__coordinates_original = template_image_coords

        rows = template_image_rectified.shape[0]
        cols = template_image_rectified.shape[1]
        assert (rows != 0)
        assert (cols != 0)
        num_pixels = rows * cols

        if len(template_image_rectified.shape) == 3:
            im_gray = cv2.cvtColor(template_image_rectified, cv2.COLOR_RGB2GRAY)
            self.__image_rectified = im_gray
        else:
            self.__image_rectified = np.copy(template_image_rectified)

#        self.__image_rectified = cv2.GaussianBlur(self.__image_rectified, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)
        if self.__equalize:
            self.__image_rectified = cv2.equalizeHist(self.__image_rectified)

        self.__gradients = np.float64(computeGrayImageGradients(self.__image_rectified))
        self.__gray_levels = np.float64(self.__image_rectified.reshape(num_pixels, 1))

        self.__changeRectifiedTemplateSize(self.__image_rectified.shape)

    def computeReferenceFeaturesGradient(self):
        """
        Computes the grey levels gradient of the rectified template image or any other feature
        in the template model.

        It computes the  \frac{\partial I(\vx)}{\partial \vx} (the gradient).
        The size of the output matrix is NxK being N the number of pixels and K=2 (cartesian
        coodinates in R2).

        :return: A np array being Nx2 (number of template pixels x 2 )
        """
        return self.__gradients

    def __resampleImage(self, image, coords, template_image_shape):
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

        # We don't do anything special with the warped image (nor DCT, nor extracting borders, etc).
        if len(image.shape) == 3:
            im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            im_gray = image

        im_cols = im_gray.shape[1]
        im_rows = im_gray.shape[0]
        #num_pixels_img = im_cols*im_rows

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

        # As template top left corner coords are are (-width/2, -height/2) then, in order
        # to get (0,0) in that corner, we have to add (width/2, height/2) to all coordinates
        # that are going to sample from an image.
        floor_x = np.floor(coords[:, 0])
        ceil_x = np.ceil(coords[:, 0])
        floor_y = np.floor(coords[:, 1])
        ceil_y = np.ceil(coords[:, 1])

        g0 = np.zeros((coords.shape[0], 1))
        g1 = np.zeros((coords.shape[0], 1))
        g2 = np.zeros((coords.shape[0], 1))
        g3 = np.zeros((coords.shape[0], 1))

        l = (floor_y >= 0) & (ceil_y < im_rows) & (floor_x >= 0) & (ceil_x < im_cols)
        im_gray_flatten = im_gray.flatten()[:, np.newaxis]
        g0[l, :] = np.float32(im_gray_flatten[np.int32(floor_x[l] + im_cols*floor_y[l])])
        g1[l, :] = np.float32(im_gray_flatten[np.int32(floor_x[l] + im_cols*ceil_y[l])])
        g2[l, :] = np.float32(im_gray_flatten[np.int32(ceil_x[l] + im_cols*ceil_y[l])])
        g3[l, :] = np.float32(im_gray_flatten[np.int32(ceil_x[l] + im_cols*floor_y[l])])

        x_delta = coords[:, 0] - floor_x[:]
        x_delta = x_delta[:, np.newaxis]
        y_delta = coords[:, 1] - floor_y[:]
        y_delta = y_delta[:, np.newaxis]

        g01 = g0 + (g1 - g0)*x_delta
        g23 = g2 + (g3 - g2)*y_delta

        features = g01 + (g23 - g01)*y_delta

        warped_img = np.uint8(np.reshape(features, (template_image_shape[0], template_image_shape[1])))

        return warped_img

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

        assert(self.__template_image_shape is not None)

        warped_img = self.__resampleImage(image, coords, self.__template_image_shape)

        # If the template image is not set in the constructor then the template image is
        # set to the gray levels of the image and coordinates of the first call to
        # computeImageFeatures
        if self.__image is None:
            self.setTemplateImage(warped_img)

#        warped_img = cv2.GaussianBlur(warped_img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)
        if self.__equalize:
            warped_img = np.float64(cv2.equalizeHist(warped_img))

        num_template_pixels = self.__image_rectified.shape[0] * self.__image_rectified.shape[1]
        features = np.reshape(warped_img, (num_template_pixels, 1))

        return features

    def getTemplateImageAndCoords(self):
        """
        Returns the image where the template is embedded. Also get the coordinates (as a numpy array (Nx2, x-y))
        where the template is in the returned image.

        :return: A tuple (image, coordinates)
        """
        return self.__image, self.__coordinates_original


    def computeReferenceFeatures(self, object_params=None):
        """
        Returns the rectified template image gray levels as a vector. Do not accept any object params in the
        case of gray images.

        :param object_params: A np array with the object params (in this case is always None).
        :return: A vector that is Nx1 (number of template pixels x 1 ).
        """

        gray_levels = np.copy(self.__gray_levels)

        return gray_levels

    def getReferenceCoords(self):
        """
        Returns the coordinates of the rectified image template points (the reference coordinates).

        Returns the coordinates of each pixel from left to right and top to
        bottom (by rows). The coordinates of the top left corner are (-width/2,-height/2),
        and the coordinates of the right bottom corner are  (width/2, height/2). That
        means that the image center pixel take template (reference) coordinates (0,0).

        :return  A np array that is Nx2 (number of template pixels/features x 2)
        """

        return np.copy(self.__coordinates_rectified)

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
        return self.self.__template_image_shape[0] * self.__template_image_shape[1]

    def convertReferenceFeaturesToImage(self, features):
        return np.uint8(np.reshape(features, (self.__template_image_shape[0], self.__template_image_shape[1])))

    def __computeRectifiedTemplateCoordinates(self, gray_image_shape):

        assert (len(gray_image_shape) < 3)

        rows = gray_image_shape[0]
        cols = gray_image_shape[1]
        width_div_2 = round(cols / 2.0)
        height_div_2 = round(rows / 2.0)

        coords_list = [[j - width_div_2, i - height_div_2] for i in range(rows) for j in range(cols)]
        coordinates = np.array(coords_list)

        return coordinates
