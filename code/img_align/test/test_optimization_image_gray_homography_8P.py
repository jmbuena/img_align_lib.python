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
from img_align.cost_functions import CostFunL2ImagesInvCompRegressor
from img_align.optimizers import OptimizerGaussNewton


class TestMotionInvCompImageGrayHomography8P(unittest.TestCase):

    def getInitialParams(self, template):

        # The template center point should be the (0,0). Therefore we need to
        # correct start the homography H in order to be the right one.
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
        H = H / H[2, 2]

        initial_params = np.copy(np.reshape(H, (9, 1)))
        initial_params = initial_params[0:8, :]

        return initial_params

    def showResults(self, frame, motion_params, object_model, motion_model):
        """
        Show the tracking results over the given frame (but not calls to cv2.waitKey())

        :param frame: plot results over this frame
        :return: The results plotted over the input frame with OpenCV commands.
        """

        ref_coords = object_model.getReferenceCoords()
        ctrl_indices, ctrl_lines = object_model.getCtrlPointsIndices()
        image_coords = np.int32(motion_model.map(ref_coords, motion_params))

        H = np.reshape(np.append(motion_params, 1.0), (3, 3))

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

    def test_inv_comp(self):

        # For generating different examples, the template should be a full image with the template
        # (i.e. book cover) embedded in the middle of the image.
        template = cv2.imread(os.path.join('resources', 'book_mp4_first_image.jpg'))
        self.assertTrue(template is not None)

        # The rectified template (fronto-parallel image of the book).
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_80x73.jpg'))
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_53x49.jpg'))
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_26x24.jpg'))
        rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template.jpg'))
        self.assertTrue(rectified_template is not None)
        if len(rectified_template.shape) == 3:
            rectified_template = cv2.cvtColor(rectified_template, cv2.COLOR_RGB2GRAY)

        initial_params = self.getInitialParams(rectified_template)

        object_model = ModelImageGray(template_image_shape=rectified_template.shape, equalize=True)
        motion_model = MotionHomography8P()

        reference_coords = object_model.getReferenceCoords()
        template_coords = motion_model.map(reference_coords, initial_params)
        object_model.setTemplateImage(template, template_coords)

        cost_function = CostFunL2ImagesInvComp(object_model, motion_model, show_debug_info=False)
        optimizer = OptimizerGaussNewton(cost_function, max_iter=30, show_iter=False)

        self.tracking(initial_params, object_model, motion_model, optimizer)

    def test_inv_comp_regressor(self):

        # For generating different examples, the template should be a full image with the template
        # (i.e. book cover) embedded in the middle of the image
        template = cv2.imread(os.path.join('resources', 'book_mp4_first_image.jpg'))
        self.assertTrue(template is not None)

        # The rectified template (fronto-parallel image of the book).
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template.jpg'))
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_80x73.jpg'))
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_53x49.jpg'))
        #rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template_26x24.jpg'))
        rectified_template = cv2.imread(os.path.join('resources', 'book_mp4_template.jpg'))
        self.assertTrue(rectified_template is not None)
        if len(rectified_template.shape) == 3:
            rectified_template = cv2.cvtColor(rectified_template, cv2.COLOR_RGB2GRAY)

        initial_params = self.getInitialParams(rectified_template)

        object_model = ModelImageGray(template_image_shape=rectified_template.shape, equalize=True)
        motion_model = MotionHomography8P()

        reference_coords = object_model.getReferenceCoords()
        template_coords = motion_model.map(reference_coords, initial_params)

        object_model.setTemplateImage(template, template_coords)

        cost_function = CostFunL2ImagesInvCompRegressor(object_model, motion_model, show_debug_info=True)
        optimizer = OptimizerGaussNewton(cost_function, max_iter=30, show_iter=False)

        self.tracking(initial_params, object_model, motion_model, optimizer)

    def tracking(self, initial_params, object_model, motion_model, optimizer):

        #video_source = os.path.join('resources', 'book1.mp4')
        video_source = os.path.join('resources', 'book1.mpeg')

        cv2.namedWindow('Video')
        video_capture = cv2.VideoCapture(video_source)
        params = initial_params
        #while True:
        for i in range(300):
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            # frame = cv2.imread('resources/00000002.jpg')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            params = optimizer.solve(gray, params)

            # Display the resulting frame
            self.showResults(frame, params, object_model, motion_model)
            cv2.imshow('Video', frame)
            #cv2.imwrite(os.path.join('resources', 'book_kk_{}.jpg'.format(i)), frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()



