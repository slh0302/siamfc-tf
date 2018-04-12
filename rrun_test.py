from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from utils.groundTracking import fetchSingleResult
from utils.preprocess import *

def _init_video(env, evaluation, video, beginframe='', size=0):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = []
    file_name = beginframe.split('_')[-1] + ".jpg"
    total = 0
    for f in os.listdir(video_folder):
        if beginframe != '' and f != file_name:
            continue
        if f.endswith(".jpg"):
            total += 1
            if total > size:
                break
            if beginframe!='':
                beginframe=''
            frame_name_list.append(f)

    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()

    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    # gt_file = os.path.join(video_folder, 'groundtruth.txt')
    # gt = np.genfromtxt(gt_file, delimiter=',')
    # n_frames = len(frame_name_list)
    # assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return frame_name_list, frame_sz

# avoid printing TF debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# TODO: allow parameters from command line or leave everything in json files?
hp, evaluation, run, env, design = parse_arguments()
# Set size for use with tf.image.resize_images with align_corners=True.
# For example,
#   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
# instead of
# [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
final_score_sz = hp.response_up * (design.score_sz - 1) + 1
# build TF graph once for all
filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)
target_id = '19'
bbox, begin = fetchSingleResult('./data/DETRAC-Train-Annotations-XML/MVI_40213.xml')
frame_name_list, _ = _init_video(env, evaluation, evaluation.video, begin[target_id][0], len(bbox[target_id]))
pos_x, pos_y, target_w, target_h = region_to_bbox(bbox[target_id][evaluation.start_frame])
print pos_x, pos_y, target_w, target_h
bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                        filename, image, templates_z, scores, evaluation.start_frame)

generateImg(frame_name_list, bboxes, "/home/slh/project/siamfc-tf/data/output1")
# _, precision, precision_auc, iou = compile_results(gt, bboxes, evaluation.dist_threshold)
# print evaluation.video + \
#       ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision + \
#       ' -- Precision AUC: ' + "%.2f" % precision_auc + \
#       ' -- IOU: ' + "%.2f" % iou + \
#       ' -- Speed: ' + "%.2f" % speed + ' --'
# print
