
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
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ImageSequence:

    def __init__(self, exp_file):
        self.exp_file = exp_file

    def load(self):
        '''
        :param exp_file: xml file with the sequence configuration
        '''

    #     if os.path.exists(self.exp_file):
    #         try:
    #             xml_tree = ET.parse(self.exp_file)
    #             xml_root = xml_tree.getroot()
    #
    #             # Parse the XML file
    #             print "Parsing XML experiment file {}".format(self.exp_file)
    #
    #             # Sequence name
    #             if xml_root.find('sequence') is not None:
    #                 self.sequence_name = xml_root.find('sequence').text
    #
    #             algorithm_xml = xml_root.find('algorithm')
    #             if algorithm_xml is not None:
    #
    #                 # Motion Model configuration.
    #                 self.motion_config = self.__parseXMLPart(algorithm_xml, 'motion_model')
    #
    #                 # Object Model configuration.
    #                 self.object_config = self.__parseXMLPart(algorithm_xml, 'object_model')
    #
    #                 # Cost Function configuration.
    #                 self.cost_config = self.__parseXMLPart(algorithm_xml, 'cost_function')
    #
    #                 # Optimizer configuration.
    #                 self.optimizer_config = self.__parseXMLPart(algorithm_xml, 'optimizer')
    #
    #                 # Concat all the dictionaries in one
    #                 self.all_config = dict(self.motion_config, **self.object_config)
    #                 self.all_config.update(self.cost_config)
    #                 self.all_config.update(self.optimizer_config)
    #
    #         except IOError as e:
    #             print e.strerror
    #
    # def __parseXMLPart(self, xml_root, section_name):
    #     config_params = dict()
    #     if xml_root.find(section_name) is not None:
    #         config_params[section_name + '_name'] = xml_root.find(section_name).find('name').text
    #         parameters = xml_root.find(section_name).find('parameters')
    #
    #         if parameters is None:
    #             parameters = []
    #
    #         for param in parameters:
    #             param_name = param.find('name').text
    #             param_type = param.find('type').text
    #             param_value = param.find('value').text
    #
    #             config_params[param_name] = param_value
    #             if param_type == 'int':
    #                 config_params[param_name] = int(param_value)
    #             elif param_type == 'float':
    #                 config_params[param_name] = float(param_value)
    #             elif param_type == 'bool':
    #                 config_params[param_name] = (param_value == 'True') or (param_value == '1')
    #
    #     return config_params

