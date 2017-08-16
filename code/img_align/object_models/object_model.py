# @brief Object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2017/08/16 (Modified)
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class ObjectModel:
    """
    A class that defines the interface for the object (target) model.
    Can be an image, an appearance model (PCA of images), a shape model
    with landmarks (e.g. an AMM), a 3D model with texture, etc.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def computeFeaturesGradient(self):
        """
        Computes the grey levels gradient of a template image or any other feature
        in the template model.

        It is the equivalent to the \frac{\partial I(\vx)}{\partial \vx} (the
        gradient) whenever we have an image as a model. The size of the output
        matrix is NxK being N the number of pixels/features and K the
        dimensionality of \vx (the template coordinates vector). For example,
        in the case of a 2D planar model dim(\vx) = 2 and in the case of a 3D
        object dim(\vx)=3

        In a given feature vector we can have C different channels (e.g. a ConvNet
        output, BitPlanes, Descriptor Fields, etc). In this way a grey level image
        has C=1, a RGB image has C=3 and an RGB-D image has C=4.

        Images are scanned in row major order (one row after another). Therefore
        pixel (0,0) is going first, (0,1) is second, (0,2) is third, etc.

        :return: A np array being NxKxC (number of points x dim(\vx) x number of channels)
        """
        return


    @abc.abstractmethod
    def computeImageFeatures(self, image):
        """
        Computes the features vector from a given image (it can be Gray, RGB, RGB-D, etc).

        This method is intended to compute the same image features as used in the
        template (Grey levels, gradient orientations, Descriptor Fields, Bit-Planes, etc)
        from the input image. If features are just plain gray levels
        then this method returns the input image grey levels in an Nx1 vector.

        Returns a matrix with a channel feature per column. The feature vector
        corresponds to a reference point on the object model. Images are scanned
        in row major order (one row after another). Therefore on each column of
        the output feature in channel c at pixel (0,0) is going
        first, (0,1) is second, (0,2) is third, etc.

        :param image: input image to compute the feature channels on.
        :return: A np array that is NxC (number of points x number of channels )
        """
        return

    @abc.abstractmethod
    def computeTemplateFeatures(self, object_params=None):
        """
        Computes the features vector from the template (the target object to be tracked)

        This method is intended to compute the same image features as used in the
        template (DCT, gradients orientations, Descriptor Fields, Bit-Planes, etc)
        from the input image. If features are just plain gray levels
        then this method returns the input image grey levels in an Nx1 vector.

        Returns a matrix with a channel feature per column. The feature vector
        corresponds to a reference point on the object model. Images are scanned
        in row major order (one row after another). Therefore on each column of
        the output feature in channel c at pixel (0,0) is going
        first, (0,1) is second, (0,2) is third, etc.

        :object_params: parameters of object configuration (e.g. PCA parameters in AAMs).
        :return: A np array that is NxC (number of points x number of channels )
        """
        return


    @abc.abstractmethod
    def getReferenceCoords(self):
        """
        reference_coords are the the coordinates of the points used to track. If we
        have a template image as object model then those are the coordinates attributed
        to every single pixel used in tracking. If we use a 3D Morphable Model those
        are the texture coordinates of every single 3D point used in tracking.

        :return  A np array that is NxK (number of template points x dim(\vx))
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

        control_points_lines are a list of tuples with the indices of reference points
        (that are control points) joint by the line.

        :return: (control_points_indices, control_points_lines)
        """
        return


    @abc.abstractmethod
    def getNumOfReferenceCoords(self):
        return