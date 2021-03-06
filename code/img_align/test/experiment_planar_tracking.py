# @brief Optimization algorithm interface.
# @author Jose M. Buenaposada
# @date 2017/10/10
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

from img_align.optimizers import OptimizerFactory
from img_align.test import ImageSequence
from img_align.test import ImageSequenceResults

class ExperimentPlanarTracking:

    def __init__(self, exp_file):
        self.exp_file = exp_file
        self.__motion_config = None
        self.__object_config = None
        self.__cost_config = None
        self.__optimizer_config = None
        self.__all_config = None
        self.__show_results = False
        self.__sequence_results_name = None
        self.__optimizer = None

        self.sequence_name = None
        self.test_name = None
        self.results_dir = None
        self.sequence_results = None
        self.sequence = None

    def open(self):
        '''
        :param exp_file: xml file with the experiment configuration
        '''

        if os.path.exists(self.exp_file):
            try:
                xml_tree = ET.parse(self.exp_file)
                xml_root = xml_tree.getroot()

                # Parse the XML file
                print "Parsing XML experiment file {}\n".format(self.exp_file)

                # test name
                if xml_root.find('experiment_name') is not None:
                    self.test_name = xml_root.find('experiment_name').text

                # Sequence name
                if xml_root.find('sequence') is not None:
                    self.__sequence_name = xml_root.find('sequence').text

                # Execution params
                if xml_root.find('execution_parameters') is not None:
                    self.results_dir = xml_root.find('execution_parameters').find('save_dir').text
                    show_results_str = xml_root.find('execution_parameters').find('show').text
                    self.__show_results = (show_results_str.upper() == "TRUE" or show_results_str == "1")

                head, tail = os.path.splitext(self.__sequence_name)
                head, sequence_name = os.path.split(head)

                self.__sequence_results_name = os.path.join(self.results_dir, self.test_name + '.results.xml')

                algorithm_xml = xml_root.find('algorithm')
                if algorithm_xml is not None:

                    # Motion Model configuration.
                    self.__motion_config = self.__parseXMLPart(algorithm_xml, 'motion_model')

                    # Object Model configuration.
                    self.__object_config = self.__parseXMLPart(algorithm_xml, 'object_model')

                    # Cost Function configuration.
                    self.__cost_config = self.__parseXMLPart(algorithm_xml, 'cost_function')

                    # Optimizer configuration.
                    self.__optimizer_config = self.__parseXMLPart(algorithm_xml, 'optimizer')

                    # Concat all the dictionaries in one
                    self.__all_config = dict(self.__motion_config, **self.__object_config)
                    self.__all_config.update(self.__cost_config)
                    self.__all_config.update(self.__optimizer_config)

            except IOError as e:
                print e.strerror

    def __parseXMLPart(self, xml_root, section_name):
        config_params = dict()
        if xml_root.find(section_name) is not None:
            config_params[section_name + '_name'] = xml_root.find(section_name).find('name').text
            parameters = xml_root.find(section_name).find('parameters')

            if parameters is None:
                parameters = []

            for param in parameters:
                param_name = param.find('name').text
                param_type = param.find('type').text
                param_value = param.find('value').text

                config_params[param_name] = param_value
                if param_type == 'int':
                    config_params[param_name] = int(param_value)
                elif param_type == 'float':
                    config_params[param_name] = float(param_value)
                elif param_type == 'bool':
                    config_params[param_name] = (param_value.upper() == 'TRUE') or (param_value == '1')
                elif param_type == 'num_list':
                    num_list = [float(x) for x in param_value.split()]
                    config_params[param_name] = np.array(num_list)

        return config_params

    def computeImageCoords(self, motion_params):
        ref_coords = self.__optimizer.cost_function.object_model.getReferenceCoords()
        ctrl_indices, ctrl_lines = self.__optimizer.cost_function.object_model.getCtrlPointsIndices()
        image_coords = self.__optimizer.cost_function.motion_model.map(ref_coords, motion_params)

        # Make a dictionary for coordinates changes. The ref_coords are all over the template and ctrl points are only
        # 4 in the case of images object models. We like to change ctrl_lines indices to move only over the ctrl points
        # in an np.ndarray of 4 points, not over all the ref_coords indices.
        dict_ctrl_points = dict()
        for i in range(len(ctrl_indices)):
            dict_ctrl_points[ctrl_indices[i]] = i

        ctrl_lines_new = list(ctrl_lines) # copy of first list (list of lists)
        for i in range(len(ctrl_lines)):
            ctrl_lines_new[i] = list(ctrl_lines[i]) # copy of i-th internal
            ctrl_lines_new[i][0] = dict_ctrl_points[ctrl_lines[i][0]]
            ctrl_lines_new[i][1] = dict_ctrl_points[ctrl_lines[i][1]]

        return image_coords[ctrl_indices], ctrl_lines_new

    def showResults(self, frame, image_coords, coords_lines, ground_truth_display=False):
        """
        Show the tracking results over the given frame

        :param frame: plot results over this frame
        :param image_coords:
        """
        if not ground_truth_display:
            line_color = (255, 255, 255)
            line_thickness = 3
            ctrl_point_color = (0, 0, 255)
            ctrl_point_radius = 4
        else:
            line_color = (0, 255, 255)
            line_thickness = 2
            ctrl_point_color = (125, 0, 125)
            ctrl_point_radius = 2

        for i in range(len(coords_lines)):
            index1 = coords_lines[i][0]
            index2 = coords_lines[i][1]

            cv2.line(frame,
                     (int(image_coords[index1, 0]), int(image_coords[index1, 1])),
                     (int(image_coords[index2, 0]), int(image_coords[index2, 1])),
                     color=line_color,
                     thickness=line_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for j in range(image_coords.shape[0]):
            cv2.circle(frame,
                       (int(image_coords[j, 0]), int(image_coords[j, 1])),
                       ctrl_point_radius,
                       ctrl_point_color,
                       -1)  # filled
            cv2.putText(frame, str(j), (int(image_coords[j, 0]), int(image_coords[j, 1])),
                        font, 1, (255, 255, 255), 2)

        return image_coords

    def run(self):
        '''
        Run the visual tracking experiment over the data-set in the xml file,
        loaded in the load method
        '''

        self.sequence = ImageSequence(self.__sequence_name)
        self.sequence.open()

        if self.__sequence_results_name is not None:
            self.sequence_results = ImageSequenceResults(self.test_name,
                                                         seq_file = self.__sequence_name,
                                                         results_file=self.__sequence_results_name)

        # if the sequence is already processed we do not run over it again.
        if os.path.exists(self.__sequence_results_name):
            self.sequence_results.open()
            return

        optimizer_factory = OptimizerFactory()
        self.__optimizer = optimizer_factory.getOptimizer(self.__all_config)

        params = None
        while self.sequence.nextFrame():
            frame, gt_corners, frame_name = self.sequence.getCurrentFrame()
            if params is None:
                # Get the template from the first image using the ground truth parameters
                template_coords = self.__optimizer.cost_function.object_model.getReferenceCoords()
                ctrl_indices, ctrl_lines = self.__optimizer.cost_function.object_model.getCtrlPointsIndices()
                template_ctrl_coords = template_coords[ctrl_indices, :]

                params = self.__optimizer.cost_function.motion_model.computeParams(template_ctrl_coords, gt_corners)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            params = self.__optimizer.solve(gray, params)
            estimated_corners, ctrl_lines = self.computeImageCoords(params)

            if self.__show_results:
                self.showResults(frame, estimated_corners, ctrl_lines)
                self.showResults(frame, gt_corners, ctrl_lines, ground_truth_display=True)

                cv2.imshow('Video', frame)
                # if cv2.waitKey(20) & 0xFF == ord('q'):
                #     break
                cv2.waitKey(20)

            if self.__sequence_results_name is not None:
                self.sequence_results.addFrameTrial(img=frame.copy(),
                                               name=frame_name,
                                               corners=estimated_corners.copy(),
                                               profiling_info=self.__optimizer.getProfilingInfo())

        self.sequence.close()
        if self.__show_results:
            cv2.destroyAllWindows()

        if self.__sequence_results_name is not None:
            self.sequence_results.write()

