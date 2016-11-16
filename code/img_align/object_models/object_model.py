# @brief Object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class ObjectModel:
    """A class that defines the interface for the object (target) model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def computeTemplateFeaturesGradient(self):
        """
        Computes the grey levels gradient of a template image or any other feature
        in the template model.

        It is the equivalent to the \frac{\partial I(\vx)}{\partial \vx} (the
        gradient) whenever we have an image as a model. The size of the output
        matrix is Nxk being N the number of pixels/features and k the
        dimensionality of \vx (the template coordinates vector). For example,
        in the case of a 2D planar model dim(\vx) = 2 and in the case of a 3D
        object dim(\vx)=3

        :return: A np array being Nxk (number of template points x dim(\vx))
        """
        return


    @abc.abstractmethod
    def extractFeaturesFromWarpedImage(self, warped_image):
        """
        Computes the features vector from the warped image

        This method is intended to compute the image features (DCT, gradients
        orientations, etc) from the warped input image (to the reference
        coordinates of the template). If features are just plain gray levels
        then this method returns the input image grey levels in an Nx1 vector.

        Returns a matrix with a feature vector per row. The feature vector
        corresponds to a reference point on the object model.

        :param warped_image:
        :return: A np array that is Nxk (number of template points x f )
        """
        return

    @abc.abstractmethod
    def computeTemplateFeatures(self, object_params):
        """
        Computes the features vector from the template (the target object to be tracked)

        Returns a matrix with a feature vector per row. Each features vector corresponds
        to a reference point in the object model.

        :param object_params: Actual motion params needed in some object models
        :return: A np array that is Nxf (number of template pixels/features x f )
        """
        return


    @abc.abstractmethod
    def getCtrlPointsIndices(self):
        """
        Returns the coordinates of template points (the reference coordinates).

        reference_coords are the the coordinates of the points used to track. If we
        have a template image as object model then those are the coordinates attributed
        to every single pixel used in tracking. If we use a 3D Morphable Model those are
        the texture coordinates of every single 3D point used in tracking.

        On the other hand, only some of the reference points are control points. In the
        case of a template image object model those can be the four corners of a rectangle.
        In the case of the Active Appearance Model only the triangle corners are control
        points. The control_points_indices are the column indices in reference_coords that
        are control points.

        control_points_indices are column indices of reference_coords for the control points

        :return: (control_points_indices, control_points_lines)
        """
        return


    @abc.abstractmethod
    def getReferenceCoords(self):
        """

        reference_coords are the the coordinates of the points used to track. If we
        have a template image as object model then those are the coordinates attributed
        to every single pixel used in tracking. If we use a 3D Morphable Model those are
        the texture coordinates of every single 3D point used in tracking.

        :return  A np array that is Nxk (number of template pixels/features x dim(\vx))
        """
        return

    @abc.abstractmethod
    def getNumOfReferenceCoords(self):
        return