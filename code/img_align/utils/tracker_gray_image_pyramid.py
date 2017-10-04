# @brief Tracker
# @author Jose M. Buenaposada
# @date 2017/10/04 (modified)
# @date 2016/11/14
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import numpy as np
import cv2
from img_align.optimizers import Optimizer

class TrackerGrayImagePyramid:

    def __init__(self, optimizer, pyramid_levels=1):
        self.__pyramid_levels = pyramid_levels
        self.__optimizer = optimizer
        self.__object_model = optimizer.cost_function.object_model
        self.__motion_model = optimizer.cost_function.motion_model
        num_params = self.__motion_model.getNumParams()
        self.__params = np.zeros((num_params, 1), dtype=np.float64)
        self.__template_coords = self.__object_model.getReferenceCoords()
        self.__ctrl_indices, self.__ctrl_lines = self.__object_model.getCtrlPointsIndices()
        info = np.finfo(dtype=np.float64)
        self.__max_cost = info.max

    def processFrame(self, frame, show_debug_info=False):
        """
        Track the target object on the current frame
        :param frame: current frame from camera
        """
        max_x = np.max(self.__template_coords[:,0])
        min_x = np.min(self.__template_coords[:,0])
        max_y = np.max(self.__template_coords[:,1])
        min_y = np.min(self.__template_coords[:,1])
        template_width = max_x - min_x + 1
        template_height = max_y - min_y + 1

        # Make the pyramid of images (multiresolution processing)
        pyramid = []
        pyramid.append(frame)

        width  = round(frame.shape[1]/2.0)
        height = round(frame.shape[0]/2.0)
        dst    = frame
        for i in range(self.__pyramid_levels):
            dst = cv2.pyrDown(dst, (width, height))
            pyramid.append(dst)
            width  = round(width/2.0)
            height = round(height/2.0)

            if show_debug_info:
                cv2.imshow('PyrDown', dst)
                cv2.waitKey()

            if (width < template_width) or (height < template_height):
                break

        scale_factor = pow(2.0, len(pyramid)-1)
        motion_params = self.__motion_model.scaleParams(self.__params, 1.0/scale_factor)

        if self.__optimizer.show_iter:
            print "============= Tracker starts processing frame\n"

        for i in range(len(pyramid)-1, 0, -1):
            if self.__optimizer.show_iter:
                print "\n==Iterations for ", pyramid[i].shape[0], "x", pyramid[i].shape[1], " pixels\n"

            motion_params = self.__optimizer.solve(pyramid[i], motion_params)

            if i > 0:
                motion_params = self.__motion_model.scaleParams(motion_params, 2.0)

        if self.__optimizer.show_iter:
             print "============= Tracker ENDS processing frame\n"

        self.__params = np.copy(motion_params)

        return

    def showResults(self, frame):
        """
        Show the tracking results over the given frame

        :param frame: plot results over this frame
        :return: The results plotted over the input frame with OpenCV commands.
        """
        image_coords = np.int32(self.__motion_model.map(self.__template_coords, self.__params))

        for i in range(len(self.__ctrl_lines)):
            index1 = self.__ctrl_lines[i][0]
            index2 = self.__ctrl_lines[i][1]

            cv2.line(frame,
                     (image_coords[index1,0], image_coords[index1,1]),
                     (image_coords[index2,0], image_coords[index2,1]),
                     color=(255, 255, 255), # white color
                     thickness=2)

        for j in range(len(self.__ctrl_indices)):
            index1 = self.__ctrl_indices[j]

            cv2.circle(frame,
                       (image_coords[index1, 0], image_coords[index1, 1]),
                       3,
                       (0, 0., 255.),  # red color
                       -1)  # filled

        return

    def setMotionParams(self, motion_params):
        self.__params = np.copy(motion_params)
        return

    def getMotionParams(self):
        return np.copy(self.__params)

    def getMotionModel(self):
        return self.__motion_model

    def setMaxCostFunctionValue(self, max_cost):
        self.__max_cost = max_cost

    def isLost(self):
        return False
