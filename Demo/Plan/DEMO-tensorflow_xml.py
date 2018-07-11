import os
import time
import re
import cv2
import numpy as np

import dicttoxml
import xmltodict
import json  # to convect collection.OrderedDict() to dict()
import lxml.etree as etree  # for xml pretty_print

"""
Need: pip3 install dicttoxml xmltodict (very small and download soon)
Reference: cricket_007, https://stackoverflow.com/questions/36021526/converting-an-array-dict-to-xml-in-python
Modify: JonYonv 1943
"""
'''JonYonv1943 2018-05-22'''


def write_xml(xml_pwd, xml_d, same_key_l, pretty_print=False):
    with open(xml_pwd, 'w') as f:  # write xml by dict()
        xml_str = dicttoxml.dicttoxml(xml_d, root=False, attr_type=False, ids=False).decode('utf-8')

        for same_key in same_key_l:
            xml_str = xml_str.replace('<%s><item>' % same_key, "<%s>" % same_key)
            xml_str = xml_str.replace('</item><item>', "</%s><%s>" % (same_key, same_key))
            xml_str = xml_str.replace('</item></%s>' % same_key, "</%s>" % same_key)

        xml_str = etree.tostring(etree.fromstring(xml_str), pretty_print=True).decode('utf-8') \
            if pretty_print else xml_str  # pretty print xml

        f.write(xml_str)
        return xml_str


class TensorFlowXmlIO(object):
    def __init__(self, dir_pwd, root='annotation'):
        self.root = root
        self.img_name = '.jpg'

        dir_pwd = dir_pwd.replace('\\', '/')
        dir_pwd = dir_pwd[:-1] if dir_pwd[-1] == '/' else dir_pwd
        dir_pwd = dir_pwd if dir_pwd[0] == '/' else "%s/%s" % (os.getcwd(), dir_pwd)
        self.dir_pwd = dir_pwd  # get complete pwd of [folder]
        self.img_path = os.path.join(self.dir_pwd, self.img_name)
        self.folder = self.dir_pwd.split('/')[-1]  # extract dir name

        self.width, self.height, self.depth = 'int()', 'int()', 3,

        self.tf_xml = {'annotation': {
            'folder': self.folder,
            'filename': self.img_name,
            'path': self.img_path,
            'source': {'database': 'Unknown'},
            'size': {'width': str(self.width), 'height': str(self.height), 'depth': str(self.depth)},
            'segmented': '0',
            'object': [],
        }}

        # self.label_name, self.score = 'label_name', 'int(0~100)'
        # self.xmin, self.ymin, self.xmax, self.ymax = 'int()', 'int()', 'int()', 'int()'
        # self.tf_obj = {
        #     'name': 'name', 'score': 'int(0~100)',
        #     'bndbox': {'xmin': self.xmin, 'ymin': self.ymax, 'xmax': self.xmax, 'ymax': self.ymax},
        #     'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        # }

    def update_img_shape(self, img_name):
        self.img_name = img_name
        self.img_path = os.path.join(self.dir_pwd, self.img_name)

        self.tf_xml[self.root]['filename'] = self.img_name
        self.tf_xml[self.root]['path'] = self.img_path

        img_shape = cv2.imread(self.img_path).shape
        # self.width, self.height = img_shape[1], img_shape[0]
        # self.depth = img_shape[2] if len(img_shape) == 3 else 1

        self.width, self.height, self.depth = img_shape[1], img_shape[0], img_shape[2] if len(img_shape) == 3 else 1
        self.tf_xml[self.root]['size'] = {'width': str(self.width), 'height': str(self.height), 'depth': str(self.depth)}

    def add_img_object(self, label_name, score, xmin, ymin, xmax, ymax):
        xmin = str(int(float(xmin) * float(self.width)))
        ymin = str(int(float(ymin) * float(self.height)))
        xmax = str(int(float(xmax) * float(self.width)))
        ymax = str(int(float(ymax) * float(self.height)))

        tf_obj = {
            'name': str(label_name),
            # 'score': str(score),
            'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
            'pose': 'Unspecified',
            'truncated': '0',
            'difficult': '0',
        }
        return tf_obj

    def csv_to_xml(self, csv_pwd):
        with open(csv_pwd, 'r') as f:
            csv_l = f.readlines()
            print("||| csv_len:", len(csv_l))

        for i, csv_result in enumerate(csv_l):
            csv_result_l = csv_result.split(',')
            img_name = csv_result_l[0]

            print(i, img_name)
            self.update_img_shape(img_name)
            self.tf_xml[self.root]['object'] = []

            tf_result_l = np.array(csv_result_l[1:]).reshape(-1, 6)
            for j, (label, score, y_min, x_min, y_max, x_max) in enumerate(tf_result_l):
                tf_obj = self.add_img_object(label, score, x_min, y_min, x_max, y_max)  # notice: order change
                self.tf_xml[self.root]['object'].append(tf_obj)

            xml_pwd = "%s.xml" % self.img_path[:-4]
            write_xml(xml_pwd, self.tf_xml, same_key_l=['object', ], pretty_print=True)


if __name__ == '__main__':
    # file_pwd = 'xml_right'
    file_pwd = '/media/vision/HD_2TB/code/Test/xml_test'
    csv_pwd = '/media/vision/HD_2TB/code/Test/xml_test/0_image_information.csv'

    tf_xml_io = TensorFlowXmlIO(file_pwd)
    tf_xml_io.csv_to_xml(csv_pwd)
