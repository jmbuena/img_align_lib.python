
# @brief Image Sequence representation of testing.
# @author Jose M. Buenaposada
# @date 2017/10/11
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ImageSequence:

    def __init__(self, seq_file):
        self.seq_file = seq_file
        self.is_video_file = False
        self.is_opened = False
        self.video_file_path = ''
        self.sequence_frames = []
        self.current_frame = None
        self.current_frame_index = 0
        self.current_corners = None

    def load(self):

        '''
        :param exp_file: xml file with the sequence configuration
        '''

        if not os.path.exists(self.seq_file):
            raise ValueError('The sequence configuration file {}, not found!'.format(self.seq_file))

        try:
            xml_tree = ET.parse(self.seq_file)
            xml_root = xml_tree.getroot()

            # Parse the XML file
            print "Parsing XML sequence file {}".format(self.seq_file)

            # Check if we are processing a video file (.avi, .mpeg, etc) ground truth
            video_file = xml_root.find('video_file')
            if video_file is not None:
                self.is_video_file = True
                self.video_file_path = video_file.text

            frames = xml_root.findall('frame')

            for frame in frames:
                # if it is a video file, then the image_name is the frame number (as text).
                image_name = frame.find('image').text
                corners = []
                if frame.find('ground_truth') is not None:
                    # Not checking for corners existence nor correctness !!
                    corners = [float(x) for x in frame.find('ground_truth').find('corners').text.split()]
                    corners = np.array(corners).reshape(4, 2)

                self.sequence_frames.append((image_name, corners))

        except IOError as e:
            print e.strerror

    def open(self):
        '''
        Open the video file it is the case. If we are dealing with an image sequence with image files on disk, this
        method does nothing.
        '''
        if self.is_video_file:
            self.video_capture = cv2.VideoCapture(self.video_file_path)

        self.current_frame_index = 0
        self.is_opened = True

    def close(self):
        '''
        Close the video file it is the case. If we are dealing with an image sequence with image files on disk, this
        method does nothing.
        '''
        if self.is_video_file and self.is_opened:
            self.video_capture.release()

        self.is_opened = False

    def nextFrame(self):
        '''
        Move to the next frame to read.
        '''
        if not self.is_opened:
            return

        frame_name, self.current_corners = self.sequence_frames[self.current_frame_index]
        if self.is_video_file:
            ret, self.current_frame = self.video_capture.read()
        else:
            self.current_frame = cv2.imread(frame_name)

        self.current_frame_index = self.current_frame_index + 1

    def getCurrentFrame(self):

        if not self.is_opened:
            return None

        return (self.current_frame,  self.current_corners)
