# @brief Object model in direct methods tracking.
# @author Jose M. Buenaposada
# @date 2017/08
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr


import unittest
import numpy as np
import cv2
import os


from img_align.motion_models import MotionHomography8P
from img_align.object_models import ModelImageGray
from img_align.cost_functions import CostFunL2ImagesInvComp
from img_align.optimizers import OptimizerGaussNewton

class TestMotionHomography8P(unittest.TestCase):

    def setUp(self):

    #    self.template = cv2.imread(os.path.join('resources', 'book_lowres.jpg'))
        self.template = cv2.imread(os.path.join('resources', 'book_mp4_template.jpg'))
        self.initial_params = self.getInitialParams(self.template)
        assert(self.template is not None)
        self.object_model = ModelImageGray(self.template, equalize=True)
        self.motion_model = MotionHomography8P()
        self.cost_function = CostFunL2ImagesInvComp(self.object_model, self.motion_model, show_debug_info=True)
        self.optimizer = OptimizerGaussNewton(self.cost_function, max_iter=100, show_iter=True)

    def getInitialParams(self, template):

        # The template center point should be the (0,0). Therefore we need to
        # correct start the homograthe homography H in order to be the right one.
        heightDiv2 = round(template.shape[0]/2.0)
        widthDiv2 = round(template.shape[1]/2.0)
        pts1 = np.array([[-widthDiv2, -heightDiv2],
                          [widthDiv2, -heightDiv2],
                          [widthDiv2, heightDiv2],
                          [-widthDiv2, heightDiv2]], dtype=np.float32)
        pts2 = np.array([[106, 69],
                         [211, 69],
                         [211, 166],
                         [106, 166]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(pts1, pts2)
        H = H / H[2,2]

        initial_params = np.copy(np.reshape(H, (9, 1)))
        initial_params = initial_params[0:8,:]

        return initial_params

    def showResults(self, frame, motion_params):
        """
        Show the tracking results over the given frame

        :param frame: plot results over this frame
        :return: The results plotted over the input frame with OpenCV commands.
        """

        ref_coords = self.object_model.getReferenceCoords()
        ctrl_indices, ctrl_lines = self.object_model.getCtrlPointsIndices()
        image_coords = np.int32(self.motion_model.map(ref_coords, motion_params))

        H = np.reshape(np.append(motion_params, 1.0), (3,3))

        for i in range(len(ctrl_lines)):
            index1 = ctrl_lines[i][0]
            index2 = ctrl_lines[i][1]

            cv2.line(frame,
                     (image_coords[index1,0], image_coords[index1,1]),
                     (image_coords[index2,0], image_coords[index2,1]),
                     color=(255, 255, 255), # white color
                     thickness=2)

        for j in range(len(ctrl_indices)):
            index1 = ctrl_indices[j]

            cv2.circle(frame,
                       (image_coords[index1, 0], image_coords[index1, 1]),
                       3,
                       (0, 0., 255.),  # red color
                       -1)  # filled

        return


    def test_inv_comp(self):

        video_source = os.path.join('resources', 'book1.mp4')

        cv2.namedWindow('Video')
        video_capture = cv2.VideoCapture(video_source)
        params = self.initial_params
        #i = 1
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #tracker.processFrame(gray)
            params = self.optimizer.solve(frame, params)
            self.showResults(frame, params)

            # Display the resulting frame
            #tracker.showResults(frame)
            cv2.imshow('Video', frame)
            #cv2.imwrite(os.path.join('resources', 'book_kk_{}.jpg'.format(i)), frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                 break

            #i = i + 1
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()



