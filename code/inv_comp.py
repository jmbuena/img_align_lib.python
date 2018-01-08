
import argparse
import cv2
import numpy as np
import math
from img_align.motion_models import MotionHomography8P
from img_align.object_models import ModelImageGray
from img_align.cost_functions import CostFunL2ImagesInvComp
from img_align.optimizers import OptimizerGaussNewton
from img_align.utils import TrackerGrayImagePyramid

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
                                   (template_points[i][1]-y)**2) for i in range(len(template_points))]
            distances = np.array(distances)
            min_dist = distances.min()
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
            cv2.circle(copy_frame,template_points[i], 5, (0, 0, 255), -1)
            cv2.putText(copy_frame, str(i), template_points[i], font, 1, (255, 255, 255), 2)
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


def warpImageWithPerspective(image, template_dims, points):
    template_width = template_dims[0]
    template_height = template_dims[1]

    # The template is a rectangular image
    src = np.array([[0, 0],
                    [template_width, 0],
                    [template_width, template_height],
                    [0, template_height]],
                    dtype=np.float32)
    dst = np.array([[points[0][0], points[0][1]],
                    [points[1][0], points[1][1]],
                    [points[2][0], points[2][1]],
                    [points[3][0], points[3][1]]],
                    dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    template_img = cv2.warpPerspective(image, H, (template_width, template_height),
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
    H2 = np.dot(H, TR)
    initial_params = np.copy(np.reshape(H2,(9,1)))
    initial_params = initial_params[0:8, :]

    return (template_img, initial_params)


def getCommandLineTemplateImg(template_dims, coords, video_source):
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
    video_capture = cv2.VideoCapture(video_source)
    cv2.namedWindow('image')
    gray = None
    ret, frame = video_capture.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print "Error in capture"
        exit()

    copy_frame = np.copy(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(coords)):
        point = (int(coords[i][0]), int(coords[i][1]))
        cv2.circle(copy_frame, point, 5, (0,0,255), -1)
        cv2.putText(copy_frame, str(i), point, font, 1, (255,255,255), 2)
    cv2.imshow("image", copy_frame)

    cv2.waitKey()

    if gray is None:
        print "Error, gray image is None"

    return warpImageWithPerspective(gray, template_dims, coords)


def getInteractiveTemplateImg(template_dims, video_source):
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

    print "\n***PLEASE***, Click 4 corners of the template in clockwise sense, points: 0, 1, 2, and 3\n"
    print "Points **CAN BE MOVED**\n"
    print "IF you have to capture another frame or advance in a video, press key 'n' or any key with \n"
    print "less than 4 points\n\n"
    video_capture = cv2.VideoCapture(video_source)
    cv2.namedWindow('image')
    gray = None
    while True:
        ret, frame = video_capture.read()
        if frame is not None:
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

    return warpImageWithPerspective(gray, template_dims, template_points)


def main(args):
    video_source = args.video_source
    if args.video_source is None:
        video_source = 0

    template_dims = [int(args.template_dims[0]), int(args.template_dims[1])]
    if args.template_corners is None:
        global template_points
        template_img, params = getInteractiveTemplateImg(template_dims, video_source)
    else:
        template_points = np.array(args.template_corners).reshape((4,2))
        template_img, params = getCommandLineTemplateImg(template_dims, template_points, video_source)

    # Setup the optimization cost function: object model, motion model and Optimizer
    motion_model = MotionHomography8P()
    object_model = ModelImageGray(template_img, equalize=True)
    cost_function = CostFunL2ImagesInvComp(object_model, motion_model, show_debug_info=False)
    optim = OptimizerGaussNewton(cost_function, max_iter=40, show_iter=False)

    # Setup the tracker that uses a pyramid to track fast motion.
    tracker = TrackerGrayImagePyramid(optimizer=optim, pyramid_levels=2)
    tracker.setMotionParams(params)

    cv2.namedWindow('Video')
    video_capture = cv2.VideoCapture(video_source)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.processFrame(gray)
        tracker.showResults(frame)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inverse Compositional tracking with gray levels test script.')
    parser.add_argument('--video_source', dest='video_source', action='store',
                        help='Video source in OpenCV format (default: 0)')
    parser.add_argument('--template_corners', dest='template_corners', nargs=8,
                        help='template corners list in "x0 y0 x1 y1 x2 y2 x3 y3" format, without quotes (default: [])')
    parser.add_argument('--template_dims', dest='template_dims', nargs=2,
                        default=[80, 80],
                        help='template dims list in "width height" format, without quotes (default: [200, 100])')

    args = parser.parse_args()

    main(args)