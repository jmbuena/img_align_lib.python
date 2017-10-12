

import argparse
import csv
import sys
import cv2
import numpy as np


def convert_warp_data_to_frame(warp_text_row):
    return []

def plot_frame_and_data(frame, corner):
    cv2.imshow('Video', frame)
    cv2.waitKey(20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert sequence .warp to .xml format.')
    parser.add_argument('--warps_file', dest='warps_file', action='store',
                        help='File .warps from Visual Tracking Dataset')
    parser.add_argument('--video_file', dest='video_file', action='store',
                        help='Video file from Visual Tracking Dataset')
    args = parser.parse_args()

    if args.warps_file is None:
        sys.exit('Error, missing watps_file argument')

    if args.video_file is not None:
        video_capture = cv2.VideoCapture(args.video_file)
        cv2.namedWindow('Video')

    with open(args.warps_file) as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                print row
                if args.video_file is not None:
                    # Capture frame-by-frame
                    ret, frame = video_capture.read()

                if (frame is None) or (len(row) == 0):
                    break

                corners = convert_warp_data_to_frame(row)
                plot_frame_and_data(frame, corners)
                # cv2.imwrite(os.path.join('resources', 'book_kk_{}.jpg'.format(i)), frame)

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(args.warps_file, reader.line_num, e))


    if args.video_file is not None:
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
