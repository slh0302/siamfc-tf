
# -*- coding: utf8 -*-
import os
import cv2
import os.path as osp
import numpy as np
import json
import codecs
import xml.etree.ElementTree as ET
from lxml import etree
from pprint import pprint
XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

def parse_xml(fpath):

    assert fpath.endswith(XML_EXT), "Unsupport file format."

    parser = etree.XMLParser(encoding=ENCODE_METHOD)
    tree = ET.parse(fpath, parser=parser)
    root = tree.getroot()

    seq = {}
    seq['sequence_name'] = root.attrib['name']   # <sequence name="MVI_20011">

    # 1st children    <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
    seq['camera_state'] = root[0].attrib['camera_state']
    seq['sence_weather'] = root[0].attrib['sence_weather']

    # 2nd children    <ignored_region>
    ignored_regions = []
    for region in root[1]:
        xmin, ymin = int(round(float(region.attrib['left']))), int(round(float(region.attrib['top'])))
        width, height = int(round(float(region.attrib['width']))), int(round(float(region.attrib['height'])))
        ignore_region = [xmin, ymin, xmin +width, ymin +height]
        ignored_regions.append(ignore_region)
    seq['ignored_regions'] = ignored_regions

    # 3rd children    <frame>
    frames = {}
    for frame in root[2:]:
        restore_frame = {}
        restore_frame['frame_id'] = frame.attrib['num']
        restore_frame['frame_name'] = 'img%05d.jpg' % int(frame.attrib['num'])
        restore_frame['frame_density'] = frame.attrib['density']

        # target_list
        targets = {}
        for target in frame[0]:
            restore_target = {}
            restore_target['target_id'] = target.attrib['id']
            # box
            xmin, ymin = int(round(float(target[0].attrib['left']))), int(round(float(target[0].attrib['top'])))
            width, height = int(round(float(target[0].attrib['width']))), int(round(float(target[0].attrib['height'])))
            restore_target['target_bbox'] = [xmin, ymin, width, height]
            # attribute
            restore_target['target_orientation'] = target[1].attrib['orientation']
            restore_target['target_speed'] = target[1].attrib['speed']
            restore_target['target_trajectory_length'] = target[1].attrib['trajectory_length']
            restore_target['target_truncation_ratio'] = target[1].attrib['truncation_ratio']
            restore_target['target_vehicle_type'] = target[1].attrib['vehicle_type']

            targets[target.attrib['id']] = restore_target

        restore_frame['targets'] = targets
        frames['%s_img%05d' % (seq['sequence_name'], int(frame.attrib['num']))] = restore_frame

    seq['frames'] = frames
    return seq


def fetchSingleResult(xmlName):
    seq = parse_xml(xmlName)
    bboxes = {}
    begin = {}
    for frame_id in seq['frames']:
        targets = seq['frames'][frame_id]['targets']
        for tid in targets:
            target_id = targets[tid]['target_id']
            target_bbox = targets[tid]['target_bbox']
            if bboxes.has_key(target_id):
                bboxes[target_id].append(target_bbox)
            else:
                bboxes[target_id] = [target_bbox]
                begin[target_id] = [frame_id]

    return bboxes, begin

if __name__ == '__main__':
    bboxes, begin = fetchSingleResult('../data/DETRAC-Train-Annotations-XML/MVI_40213.xml')
#     bboxes = {}
#     begin = {}
#     for frame_id in seq['frames']:
#         targets = seq['frames'][frame_id]['targets']
#         for tid in targets:
#             target_id = targets[tid]['target_id']
#             target_bbox = targets[tid]['target_bbox']
#             if bboxes.has_key(target_id):
#                 bboxes[target_id].append(target_bbox)
#             else:
#                 bboxes[target_id] = [target_bbox]
#                 begin[target_id] = [frame_id]
#             # print(frame_id, target_id, target_bbox)
#
#             # break
#         # break
    print bboxes
    print begin