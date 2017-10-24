
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
import errno
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom


class ImageSequenceResults:

    def __init__(self, test_name, seq_file, results_file):
        self.test_name = test_name
        self.seq_results_file = results_file
        self.seq_file = seq_file
        self.frames = []
        self.frame_names = []
        self.frame_corners = []
        self.frame_profiling = []

    def write(self):
        '''
        '''

        try:
            xml_root = ET.Element('results')
            xml_test = ET.SubElement(xml_root, 'test')
            xml_test.text = self.test_name
            xml_sequence = ET.SubElement(xml_root, 'sequence')
            xml_sequence.text = self.seq_file
#            xml_config = ET.SubElement(xml_root, 'config')

            for i in range(len(self.frames)):
                xml_f = ET.SubElement(xml_root, 'frame')
                xml_id = ET.SubElement(xml_f, 'id')
                xml_id.text = str(i)
                xml_seq_id = ET.SubElement(xml_f, 'sequence_id')
                xml_seq_id.text = self.frame_names[i]

                xml_trial = ET.SubElement(xml_f, 'trial')
                xml_trial_id = ET.SubElement(xml_trial, 'id')
                xml_trial_id.text = str(1)
                xml_trial_corners = ET.SubElement(xml_trial, 'corners')
                corners = self.frame_corners[i]
                corners_str = "".join([" {}".format(corners[y, x]) for y in range(corners.shape[0]) for x in range(corners.shape[1])])
                xml_trial_corners.text = corners_str

                # Profiling info
                xml_trial_profiling = ET.SubElement(xml_trial, 'profiling')
                profiling_info = self.frame_profiling[i]
                xml_iter_costs = ET.SubElement(xml_trial_profiling, 'iter_costs')
                iter_costs = profiling_info['iter_costs']
                xml_iter_costs.text = "".join([" {}".format(iter_costs[j]) for j in range(len(iter_costs))])

                xml_iter_time = ET.SubElement(xml_trial_profiling, 'iter_time')
                iter_time = profiling_info['iter_time']
                xml_iter_time.text = "".join([" {}".format(iter_time[j]) for j in range(len(iter_time))])

                xml_iter_gradient_norm = ET.SubElement(xml_trial_profiling, 'iter_gradient_norm')
                iter_gradient_norm = profiling_info['iter_gradient_norm']
                xml_iter_gradient_norm.text = "".join([" {}".format(iter_gradient_norm[j]) for j in range(len(iter_gradient_norm))])

            # Create results_dir in depth
            path, file_name = os.path.split(self.seq_results_file)
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else:
                    raise

            self.__write_xml_to_file(xml_root, self.seq_results_file)

        except IOError as e:
            print str(e)

    def __write_xml_to_file(self, xml_element, file_name):
        xml_string = ET.tostring(xml_element, 'utf-8')
        parsed_string = minidom.parseString(xml_string)
        xml_string_pretty = parsed_string.toprettyxml(indent="  ")

        fid = open(file_name, 'w')
        fid.write(xml_string_pretty)
        fid.close()

    def addFrame(self, frame, name, corners, profiling_info):
        '''
        '''

        self.frames.append(frame)
        self.frame_names.append(name)
        self.frame_corners.append(corners)
        self.frame_profiling.append(profiling_info)
