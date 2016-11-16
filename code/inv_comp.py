
import cv2
import numpy as np
import math
from img_align.motion_models import Homography2DInvComp
from img_align.object_models import SingleImageModel
from img_align.optimization_problems import Homography2DInvCompProblem
from img_align.optimizers import GaussNewtonOptimizer
from img_align import Tracker

template_points = []
point_to_add = None
point_index = True
moving_point = False


def on_mouse_event(event, x, y, flags, frame):
    # grab references to the global variables
    global template_points, point_to_add, point_index, moving_point

    # if the left mouse button was clicked, record the
    # (x, y) coordinates if it is and already added point, then
    #  move it.

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(template_points) > 1:
            distances = [math.sqrt((template_points[i][0]-x)**2 +
                                   (template_points[i][1]-y)**2)  for i in range(len(template_points))]
            distances = np.array(distances)
            min_dist  = distances.min()
            if min_dist < 10:
                 point_index = distances.argmin()
                 moving_point = True

        point_to_add = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if moving_point:
            template_points[point_index] = (x, y)

        font = cv2.FONT_HERSHEY_SIMPLEX
        copy_frame = np.copy(frame)
        for i in range(len(template_points)):
            cv2.circle(copy_frame,template_points[i],5,(0,0,255),-1)
            cv2.putText(copy_frame, str(i),template_points[i], font, 1, (255,255,255),2)
        cv2.imshow("image", copy_frame)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        if moving_point:
            template_points[point_index] = (x, y)
            moving_point = False
        elif len(template_points) < 4:
            template_points.append(point_to_add)

        copy_frame = np.copy(frame)
        for i in range(len(template_points)):
            cv2.circle(copy_frame,template_points[i],5,(0,0,255),-1)
        cv2.imshow("image", copy_frame)


def getTemplateImgHomography(template_width, template_height):
    """
    Capture first image and get the template image from it:
       - Click on 4 corners of a rectangle (can be moved with mouse):
           * Point 0 should be top-left corner
           * Point 1 should be top-right corner
           * Point 2 should be lower-right corner
           * Point 3 should be lower-left corner

       - If the frame is not a nice one ... type 'n' key to get next frame
       - Any other key gets the warped template in shape=(template_img_height, template_img_width)
    :param template_width:
    :param template_height:
    :return:
    """

    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    gray = None
    while True:
        ret, frame = video_capture.read()
        if (frame is not None):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('image', frame)
            cv2.setMouseCallback('image', on_mouse_event, frame)

        else:
            print "Error in capture"
            exit()

        key = cv2.waitKey() & 0xFF
        if (key != ord('n')) and (len(template_points) == 4):
             break

    if gray is None:
        print "Error, gray image is None"


    # The template is a rectangular image
    src = np.array([[0, 0],
                    [template_width, 0],
                    [template_width, template_height],
                    [0, template_height]],
                    dtype=np.float32)
    dst = np.array([[template_points[0][0], template_points[0][1]],
                    [template_points[1][0], template_points[1][1]],
                    [template_points[2][0], template_points[2][1]],
                    [template_points[3][0], template_points[3][1]]],
                    dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    template_img = cv2.warpPerspective(gray, H, (template_width, template_height),
                                       flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)

    cv2.namedWindow('template')
    cv2.imshow('template', template_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # The template center point should be the (0,0). Therefore we need to
    # correct the homography H in order to be the right one.
    TR = np.array([[1., 0., template_width/2.],
                   [0., 1., template_height/2.],
                   [0., 0., 1.]])
    H2 = np.dot(H, TR) - np.eye(3,3)
    initial_params = np.copy(np.reshape(H2.T,(9,1)))

    return (template_img, initial_params)


def main():

    # Setup template dimensions after warping back from the first image of the sequence.
    template_img_width  = 100
    template_img_height = 200
    global template_points

    template_img, params = getTemplateImgHomography(template_img_width, template_img_height)

    # Setup the optimization problem: object model, motion model and Optimizer
    motion_model = Homography2DInvComp()
    object_model = SingleImageModel(template_img, equalize=True)
    optim_problem = Homography2DInvCompProblem(object_model, motion_model)
    optim = GaussNewtonOptimizer(optim_problem, max_iter=20, show_iter=False)

    # Setup the tracker that uses a pyramid to track fast motion.
    tracker = Tracker(optimizer=optim, pyramid_levels=1)
    tracker.setMotionParams(params)

    cv2.namedWindow('Video')
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.processFrame(gray)

        # Display the resulting frame
        tracker.showResults(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
             break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()