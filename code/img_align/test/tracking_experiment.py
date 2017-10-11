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
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# import img_align.object_models
# import img_align.motion_models
# import img_align.cost_functions
import img_align.optimizers

class TrackingExperiment:

    def __init__(self, exp_file):
        self.exp_file = exp_file
        self.motion_config = None
        self.object_config = None
        self.cost_config = None
        self.optimizer_config = None
        self.all_config = None
        self.sequence_name = None

    def load(self):
        '''
        :param exp_file: xml file with the experiment configuration
        '''

        if os.path.exists(self.exp_file):
            try:
                xml_tree = ET.parse(self.exp_file)
                xml_root = xml_tree.getroot()

                # Parse the XML file
                print "Parsing XML experiment file {}".format(self.exp_file)

                # Sequence name
                if xml_root.find('sequence') is not None:
                    self.sequence_name = xml_root.find('sequence').text

                algorithm_xml = xml_root.find('algorithm')
                if algorithm_xml is not None:

                    # Motion Model configuration.
                    self.motion_config = self.__parseXMLPart(algorithm_xml, 'motion_model')

                    # Object Model configuration.
                    self.object_config = self.__parseXMLPart(algorithm_xml, 'object_model')

                    # Cost Function configuration.
                    self.cost_config = self.__parseXMLPart(algorithm_xml, 'cost_function')

                    # Optimizer configuration.
                    self.optimizer_config = self.__parseXMLPart(algorithm_xml, 'optimizer')

                    # Concat all the dictionaries in one
                    self.all_config = dict(self.motion_config, **self.object_config)
                    self.all_config.update(self.cost_config)
                    self.all_config.update(self.optimizer_config)

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
                    config_params[param_name] = (param_value == 'True') or (param_value == '1')

        return config_params

    def run(self):
        '''
        Run the visual tracking experiment over the data-set in the xml file,
        loaded in the load method
        '''

#        self.img_sequence = ImageSequence(self.sequence_name)
#        self.img_sequence.load()

#        object_model = ObjectModelFactory.getObjectModel(self.object_model_name, self.object_config)
#        motion_model = MotionModelFactory.getMotionModel(self.motion_model_name, self.motion_config)
#        cost_function = CostFunctionFactory.getCostFunction(self.motion_model_name, self.all_config)
        self.optimizer = OptimizerFactory.getOptimizer(self.all_config['optimizer_name'], self.all_config)

        # video_source = os.path.join('resources', 'book1.mp4')
        #
        # cv2.namedWindow('Video')
        # video_capture = cv2.VideoCapture(video_source)
        # params = self.initial_params
        # #i = 1
        # while True:
        #     # Capture frame-by-frame
        #     ret, frame = video_capture.read()
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        #     #tracker.processFrame(gray)
        #     params = self.optimizer.solve(frame, params)
        #     self.showResults(frame, params)
        #
        #     # Display the resulting frame
        #     #tracker.showResults(frame)
        #     cv2.imshow('Video', frame)
        #     #cv2.imwrite(os.path.join('resources', 'book_kk_{}.jpg'.format(i)), frame)
        #
        #     if cv2.waitKey(20) & 0xFF == ord('q'):
        #          break
        #
        #     #i = i + 1
        # # When everything is done, release the capture
        # video_capture.release()
        # cv2.destroyAllWindows()


