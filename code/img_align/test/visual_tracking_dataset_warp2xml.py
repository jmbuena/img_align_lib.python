

import argparse
import os
import csv
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom


class GroundTruthConstants:

    def __init__(self):
        # From file groundtruth_coordinateframe.h (https://ilab.cs.ucsb.edu/tracking_dataset_ijcv/)

        #  3D "world"  coordinates
        self.paper_width = 11.0  # inch
        self.paper_height = 8.5  # inch

        self.marker_margin = 1.0  # margin between markers and paper, inch

        self.dist_markers_x = self.paper_width + 2.0 * self.marker_margin  # inch
        self.dist_markers_y = self.paper_height + 2.0 * self.marker_margin  # inch

        self.outer_margin = 0.5 # margin around the markers, inch

        # in mm (3D points by rows)
        self.corners_world_coordinates = np.array([
            [-165.1, -133.35, 0.0],
            [165.1, -133.35, 0.0],
            [165.1, 133.35, 0.0],
            [-165.1, 133.35, 0.0]
        ])

        # warped 2D image coordinates
        self.DPI = 25.4  # i.e. 1 pixel = 1 mm
        self.st_w = int((self.dist_markers_x + 2.0 * self.outer_margin) * self.DPI)  # size of warped image
        self.dst_h = int((self.dist_markers_y + 2.0 * self.outer_margin) * self.DPI)

        x1 = float(self.outer_margin * self.DPI)
        x2 = float((self.outer_margin + self.dist_markers_x) * self.DPI)
        y1 = float(self.outer_margin * self.DPI)
        y2 = float((self.outer_margin + self.dist_markers_y) * self.DPI)

        # coordinates of where the corners should be warped to (2D points by rows)
        self.dst_corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            ])

        #self.dst_corners_center = np.array([0.5*(x2-x1), 0.5*(y2-y1)])

        # area of paper
        self.paper_margin = 0.1  # inner margin(inch)
        planarROI_x1 = (self.outer_margin + self.marker_margin + self.paper_margin) * self.DPI
        planarROI_y1 = (self.outer_margin + self.marker_margin + self.paper_margin) * self.DPI
        planarROI_x2 = (self.outer_margin + self.marker_margin + self.paper_width - self.paper_margin) * self.DPI
        planarROI_y2 = (self.outer_margin + self.marker_margin + self.paper_height - self.paper_margin) * self.DPI

        self.planarROI = np.array([
            [planarROI_x1, planarROI_y1],
            [planarROI_x2, planarROI_y1],
            [planarROI_x2, planarROI_y2],
            [planarROI_x1, planarROI_y2]
        ])

        # area of texture allowed to use
        self.texture_margin = 1.5  # inner margin(inch)

        textureROI_x1 = (self.outer_margin + self.marker_margin + self.texture_margin) * self.DPI
        textureROI_y1 = (self.outer_margin + self.marker_margin + self.texture_margin) * self.DPI
        textureROI_x2 = (self.outer_margin + self.marker_margin + self.paper_width - self.texture_margin) * self.DPI
        textureROI_y2 = (self.outer_margin + self.marker_margin + self.paper_height - self.texture_margin) * self.DPI

        self.textureROI = np.array([
            [textureROI_x1, textureROI_y1],
            [textureROI_x2, textureROI_y1],
            [textureROI_x2, textureROI_y2],
            [textureROI_x1, textureROI_y2]
        ])

        # Intrinsics matrix
        self.K = np.array([[869.57, 0, 299.748],
                           [0, 867.528, 237.284],
                           [0, 0, 1]])

        # Distortion coefficients k1, k2, p1, p2, k3
        self.distort_coeffs = np.array([-0.0225415, -0.259618, 0.00320736, -0.000551689, 0.000000000])


def convert_warp_data_to_frame(warp_text_row, gt_constants):

    row_list = warp_text_row[0].split()
    row_array = np.array([float(e) for e in row_list])

    # describe the homography that warps the actual video frame to the canonical reference frame
    H =  np.reshape(row_array, (3, 3))
    invH = np.linalg.inv(H)

    # Correction needed to get the right corners in the bricks sequences.
    T = np.array([[1.0, 0.0, 20.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  ])
    invH = np.dot(T, invH)

    coords_rows = gt_constants.dst_corners.shape[0]
    homog_coords = np.ones((coords_rows, 3), dtype=np.float64)
    #homog_coords[:, 0:2] = gt_constants.dst_corners
    homog_coords[:, 0:2] = gt_constants.textureROI
    #homog_coords[:, 0:2] = gt_constants.planarROI
    homog_new_coords = np.dot(homog_coords, invH.T)

    # Divide by the third homogeneous coordinates to get the cartesian coordinates.
    third_coord = homog_new_coords[:, 2]
    homog_new_coords = np.copy(homog_new_coords / third_coord[:, np.newaxis])

    return homog_new_coords[:, 0:2]


def plot_frame_and_data(frame, corners, gt_constants):

    image_coords = np.int32(corners)
    image_coords = np.vstack((image_coords, image_coords[image_coords.shape[0]-1, :]))

    frame_copy = frame.copy()

    for i in range(5):
        next_i = (i + 1) % 5
        cv2.line(frame_copy,
                 (image_coords[i, 0], image_coords[i, 1]),
                 (image_coords[next_i, 0], image_coords[next_i, 1]),
                 color=(255, 255, 255),  # white color
                 thickness=2)

    for j in range(4):
        cv2.circle(frame_copy,
                   (image_coords[j, 0], image_coords[j, 1]),
                   3,
                   (0, 0., 255.),  # red color
                   -1)  # filled

    cv2.imshow('Video', frame_copy)
    cv2.waitKey(20)


def undistort_frame(frame, gt_constants):
    undistorted = cv2.undistort(frame, gt_constants.K, gt_constants.distort_coeffs)

    # cv2.imshow('Video', frame)
    # cv2.imshow('Undistort', undistorted)
    #
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    #
    # diff = np.float32(gray_frame) - np.float32(gray_undistorted)
    # max_ = diff.max()
    # min_ = diff.min()
    # diff = np.uint8(255 * ((diff - min_) / (max_ - min_)))
    # cv2.imshow('Diff', diff)
    #
    # cv2.waitKey()

    return undistorted


def write_xml_to_file(xml_element, file_name):
    xml_string = ET.tostring(xml_element, 'utf-8')
    parsed_string = minidom.parseString(xml_string)
    xml_string_pretty = parsed_string.toprettyxml(indent="  ")

    fid = open(file_name, 'w')
    fid.write(xml_string_pretty)
    fid.close()


def generate_sequence_ground_truth_xml(video_file, warps_file, imgs_save_path, xml_sequence_file):

    if video_file is not None:
        video_capture = cv2.VideoCapture(video_file)
        cv2.namedWindow('Video')

    gt_constants = GroundTruthConstants()
    xml_root = ET.Element('sequence')

    img_index = 0
    with open(warps_file) as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                if video_file is not None:
                    # Capture frame-by-frame
                    ret, frame = video_capture.read()

                if (frame is None) or (len(row) == 0):
                    break

                # Read ground truth corners and display them
                frame = undistort_frame(frame, gt_constants)
                corners = convert_warp_data_to_frame(row, gt_constants)
                plot_frame_and_data(frame, corners, gt_constants)

                # Save frame
                save_img_path = os.path.join(imgs_save_path, "{0:05d}.png".format(img_index))
                cv2.imwrite(save_img_path, frame)

                # Save ground truth corners in the .xml structure to save.
                xml_frame = ET.SubElement(xml_root, 'frame')
                xml_img = ET.SubElement(xml_frame, 'image')
                xml_img.text = save_img_path
                xml_gt = ET.SubElement(xml_frame, 'ground_truth')
                xml_corners = ET.SubElement(xml_gt, 'corners')
                corners_str = " ".join([" {}".format(element) for row in corners for element in row])
                xml_corners.text = "\n" + corners_str + "\n"

                img_index = img_index + 1

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(warps_file, reader.line_num, e))

    write_xml_to_file(xml_root, xml_sequence_file)

    if video_file is not None:
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Convert sequence .warp to .xml format.')
    # parser.add_argument('--warps_file', dest='warps_file', action='store',
    #                     help='File .warps from Visual Tracking Dataset')
    # parser.add_argument('--video_file', dest='video_file', action='store',
    #                     help='Video file from Visual Tracking Dataset')
    # args = parser.parse_args()
    #
    # if args.warps_file is None:
    #     sys.exit('Error, missing warps_file argument')
    #
    # print args.video_file

    video_files = [
                   'fi-br-ld.avi',
                   'fi-br-ls.avi',
                   'fi-br-m1.avi',
                   'fi-br-m2.avi',
                   'fi-br-m3.avi',
                   'fi-br-m4.avi',
                   'fi-br-m5.avi',
                   'fi-br-m6.avi',
                   'fi-br-m7.avi',
                   'fi-br-m8.avi',
                   'fi-br-m9.avi',
                   'fi-br-pd.avi',
                   'fi-br-pn.avi',
                   'fi-br-rt.avi',
                   'fi-br-uc.avi',
                   'fi-br-zm.avi'
                  ]

    video_files_paths = [os.path.join('resources',
                                      'visual_tracking_dataset',
                                      'bricks',
                                      'videos',
                                      f) for f in video_files]

    warps_files_paths = [os.path.join('resources',
                                      'visual_tracking_dataset',
                                      'bricks',
                                      'ground_truth',
                                      f + '.warps') for f in video_files]

    for vfp, wfp in zip(video_files_paths, warps_files_paths):

        video_file_path, video_file_name = os.path.split(vfp)
        print 'Processing video file {}'.format(video_file_name)
        path_to_save_imgs = os.path.join(video_file_path, video_file_name + '.images')
        try:
            os.mkdir(path_to_save_imgs)
        except Exception as e:
            print str(e)

        warps_file_path, warps_file_name = os.path.split(wfp)
        xml_sequence_file = os.path.join(warps_file_path, warps_file_name + '.sequence.xml')

        generate_sequence_ground_truth_xml(vfp, wfp, path_to_save_imgs, xml_sequence_file)