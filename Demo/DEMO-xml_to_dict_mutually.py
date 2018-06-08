import os
import time

import cv2

import xmltodict
import dicttoxml
import json
import lxml.etree as etree

"""
Need: pip3 install dicttoxml xmltodict (very small and download soon)

Reference: cricket_007, https://stackoverflow.com/questions/36021526/converting-an-array-dict-to-xml-in-python
Midify: JonYonv 1943
"""

filename = 'lena.jpg'
img = cv2.imread(filename)

root = 'custom_root_name'
xml_dict = {root: {
    'KEY1': 1, 'KEY2': 2,
    'KEY3': {'KEY3_1': 31, 'KEY3_2': 32},
    'SAME_KEY': [],
}}

for i in range(3):
    xml_dict[root]['SAME_KEY'].append({
        'item_str': str(i), 'item2_int': int(i), 'item3_float': float(i),
        'item_list': [i, i],
        'item_dict': {i: i, i + 1: i + 1},
    })

with open('%s.xml' % filename, 'w') as f:  # write xml by dict()
    xml_bin = dicttoxml.dicttoxml(xml_dict, root=False, attr_type=False)
    f.write(xml_bin.decode('utf-8'))
    # f.write(etree.tostring(etree.fromstring(xml_bin), pretty_print=True).decode('utf-8'))  # pretty print xml
    print(xml_bin.decode('utf-8'))

with open('%s.xml' % filename, 'r') as f:  # read xml as dict()
    xml_str = ''.join([l for l in f.readlines()])
    xml_dict = dict(json.loads(json.dumps(xmltodict.parse(xml_str))))
    print(xml_dict)
