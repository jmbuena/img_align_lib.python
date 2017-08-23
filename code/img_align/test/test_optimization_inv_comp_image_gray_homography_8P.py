import unittest
import numpy as np
import cv2
import os


from img_align.motion_models import MotionHomography8P
from img_align.object_models import ModelImageGray
from img_align.cost_functions import CostFunL2ImagesInvComp

class TestMotionHomography8P(unittest.TestCase):
    None

    # def setUp(self):
    #
    #     template = cv2.imread(os.path.join('resources', 'book_mp4_template.jpg'))
    #     self.initial_params = getInitialParams(template)
    #     assert(template is not None)
    #     template_model = ModelImageGray(template, equalize=True)
    #     motion_model = MotionHomography8P()
    #     cost_function = CostFunInvCompImages(motion_model, template_model)
    #     self.optimizer = OptimizerGaussNewton(cost_function, max_iter=20, show_iter=False)
    #
    # #     # Setup the optimization problem: object model, motion model and Optimizer
    # #     motion_model = Homography2DInvComp()
    # #     object_model = SingleImageModel(template_img, equalize=True)
    # #     optim_problem = Homography2DInvCompProblem(object_model, motion_model, show_debug_info=False)
    # #     optim = GaussNewtonOptimizer(optim_problem, max_iter=20, show_iter=False)

    # def getInitialParams(self, template):
    #
    #     # The template is a rectangular image
    #     template_width, template_height = template.shape
    #     src = np.array([[0, 0],
    #                     [template_width, 0],
    #                     [template_width, template_height],
    #                     [0, template_height]],
    #                    dtype=np.float32)
    #     dst = np.array([[106, 166],
    #                     [106, 69],
    #                     [211, 69],
    #                     [211, 166]],
    #                    dtype=np.float32)
    #
    #     H = cv2.getPerspectiveTransform(src, dst)
    #
    #     # The template center point should be the (0,0). Therefore we need to
    #     # correct the homography H in order to be the right one.
    #     TR = np.array([[1., 0., template_width / 2.],
    #                    [0., 1., template_height / 2.],
    #                    [0., 0., 1.]])
    #     H2 = np.dot(H, TR) - np.eye(3, 3)
    #     initial_params = np.copy(np.reshape(H2, (9, 1)))
    #     initial_params = initial_params[0:8,:]
    #
    #     return initial_params

    # def test_inv_comp(self):
    #
    #     video_source = os.path.join('resources', 'book1.mp4')
    #
    #     cv2.namedWindow('Video')
    #     video_capture = cv2.VideoCapture(video_source)
    #     params = self.initial_params
    #     #i = 1
    #     while True:
    #         # Capture frame-by-frame
    #         ret, frame = video_capture.read()
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #         #tracker.processFrame(gray)
    #         params = self.optimizer.solve(frame, params)
    #
    #         # Display the resulting frame
    #         #tracker.showResults(frame)
    #         cv2.imshow('Video', frame)
    #         #cv2.imwrite(os.path.join('resources', 'book_kk_{}.jpg'.format(i)), frame)
    #
    #         if cv2.waitKey(10) & 0xFF == ord('q'):
    #             break
    #
    #         #i = i + 1
    #     # When everything is done, release the capture
    #     video_capture.release()
    #     cv2.destroyAllWindows()



