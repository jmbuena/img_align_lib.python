
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
import xml.etree.ElementTree as ET
from xml.dom import minidom


class FrameTrials:

    def __init__(self):
        self.image = None # gray levels or RGB
        self.name = None
        self.corners = []
        self.profiling = []


class ImageSequenceResults:

    def __init__(self, test_name, seq_file, results_file):
        self.test_name = test_name
        self.seq_results_file = results_file
        self.seq_file = seq_file
        self.__frames = dict()

    def write(self):
        try:
            xml_root = ET.Element('results')
            xml_test = ET.SubElement(xml_root, 'test')
            xml_test.text = self.test_name
            xml_sequence = ET.SubElement(xml_root, 'sequence')
            xml_sequence.text = self.seq_file
#            xml_config = ET.SubElement(xml_root, 'config')

            i = 0
            for key, frame in self.__frames.iteritems():
                xml_f = ET.SubElement(xml_root, 'frame')
                xml_id = ET.SubElement(xml_f, 'id')
                xml_id.text = str(i)
                xml_seq_id = ET.SubElement(xml_f, 'sequence_id')
                xml_seq_id.text = frame.name

                for j in range(len(frame.corners)):
                    xml_trial = ET.SubElement(xml_f, 'trial')
                    xml_trial_id = ET.SubElement(xml_trial, 'id')
                    xml_trial_id.text = str(j)
                    xml_trial_corners = ET.SubElement(xml_trial, 'corners')
                    corners = frame.corners[j]
                    corners_str = "".join([" {}".format(corners[y, x]) for y in range(corners.shape[0]) for x in range(corners.shape[1])])
                    xml_trial_corners.text = corners_str

                    # Profiling info
                    xml_trial_profiling = ET.SubElement(xml_trial, 'profiling')
                    profiling_info = frame.profiling[j]
                    xml_iter_costs = ET.SubElement(xml_trial_profiling, 'iter_costs')
                    iter_costs = profiling_info['iter_costs']
                    xml_iter_costs.text = "".join([" {}".format(iter_costs[k]) for k in range(len(iter_costs))])

                    xml_iter_time = ET.SubElement(xml_trial_profiling, 'iter_time')
                    iter_time = profiling_info['iter_time']
                    xml_iter_time.text = "".join([" {}".format(iter_time[k]) for k in range(len(iter_time))])

                    xml_iter_gradient_norm = ET.SubElement(xml_trial_profiling, 'iter_gradient_norm')
                    iter_gradient_norm = profiling_info['iter_gradient_norm']
                    xml_iter_gradient_norm.text = "".join([" {}".format(iter_gradient_norm[k]) for k in range(len(iter_gradient_norm))])

                i = i + 1

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

    def open(self):
        self.__frames = dict()

        if os.path.exists(self.seq_results_file):
            try:
                xml_tree = ET.parse(self.seq_results_file)
                xml_root = xml_tree.getroot()

                # Parse the XML file
                print "Parsing XML experiment file {}\n".format(self.seq_results_file)

                if xml_root.find('test') is not None:
                    self.test_name = xml_root.find('test').text

                if xml_root.find('sequence') is not None:
                    self.seq_file = xml_root.find('sequence').text

                frames = xml_root.findall('frame')

                for xml_frame in frames:
                    # if it is a video file, then the image_name is the frame number (as text).
                    img_id = xml_frame.find('id').text
                    seq_id = xml_frame.find('sequence_id').text

                    xml_trials = xml_frame.findall('trial')
                    for xml_trial in xml_trials:
                        trial_id = xml_trial.find('id').text
                        trial_corners = xml_trial.find('corners').text
                        corners = [float(x) for x in trial_corners.split()]
                        corners = np.array(corners).reshape(4, 2)

                        # Profiling info
                        if xml_trial.find('profiling') is not None:
                            profiling_info = dict()
                            xml_trial_profiling = xml_trial.find('profiling')

                            iter_costs_str = xml_trial_profiling.find('iter_costs').text
                            costs = [float(x) for x in iter_costs_str.split()]

                            iter_time_str = xml_trial_profiling.find('iter_time').text
                            time = [float(x) for x in iter_time_str.split()]

                            iter_gradient_norm = xml_trial_profiling.find('iter_gradient_norm').text
                            grad_norm = [float(x) for x in iter_gradient_norm.split()]

                            profiling_info['iter_costs'] = costs
                            profiling_info['iter_time'] = time
                            profiling_info['iter_gradient_norm'] = grad_norm

                            self.addFrameTrial(None, seq_id, corners, profiling_info)

            except IOError as e:
                print str(e)

    def addFrameTrial(self, img, name, corners, profiling_info):
        frame = self.__frames.get(name)
        if frame is None:
            frame = FrameTrials()
            self.__frames[name] = frame

        frame.image = img
        frame.name = name
        frame.corners.append(corners)
        frame.profiling.append(profiling_info)

    def getFrameTrials(self, name):
        return self.__frames.get(name)
