#!/usr/bin/env python2
#
# Original File: infer_simple.py from tools/
#
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##############################################################################
#
# Modification (TCA) - Add capability to save keypoints and boxes as well as
# other timing information to an output log file, as well as suppress
# visualization
#

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
from datetime import datetime
import glob
import logging
import os
import sys
import time

import pandas as pd

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--no-draw',
        dest='no_draw',
        default=False,
        action='store_true',
        help='Flag for whether to draw images or not'
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    configs_name = args.cfg.split('/')[-1][:-5]
    im_out_dir = os.path.join(args.output_dir, configs_name)
    if os.path.exists(im_out_dir):
        logger.warning('Directory {} exists...'.format(im_out_dir))
    else:
        logger.warning('Making directory: {}'.format(im_out_dir))
        os.makedirs(im_out_dir)        
        
    output_log = []
    BOX_THRESH = 0.7
    KEY_THRESH = 2
    save_threshold = True
    for idx, im_name in enumerate(im_list):
        basename = os.path.basename(im_name)
        out_name = os.path.join(
            im_out_dir, '{}'.format(basename.split('.')[0] + '.pdf'))
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers)

        time_infer = '{:.4f}'.format(time.time() - t)
        time_bbox = '{:.4f}'.format(timers['im_detect_bbox'].average_time)
        time_keypoints = '{:.4f}'.format(timers['im_detect_keypoints'].average_time)
        logger.info('Inference time: {}s'.format(time_infer))
        
        if not args.no_draw:
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                im_out_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=BOX_THRESH,
                kp_thresh=KEY_THRESH
            )
            
        boxes = cls_boxes[1]
        keypoints = cls_keyps[1]

        if save_threshold:
            # Only save keypoints and boxes that passed threshold:
            new_boxes = []
            new_keyps = []
            for ibox, box in enumerate(boxes):
                if box[-1] < BOX_THRESH:
                    continue
                new_boxes.append(box)
                new_keyps.append(keypoints[ibox])
            boxes = new_boxes
            keypoints = new_keyps
            
        output_log.append({
            "filename": im_name,
            "time_bbox": float(time_bbox),
            "time_keypoint": float(time_keypoints),
            "time_infer": float(time_infer),
            "boxes": boxes,
            "keypoints": keypoints
        })
        logger.info('Processed file {}...'.format(idx+1))

    dt_stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    model_out_filename = '{}_output_{}.json'.format(configs_name, dt_stamp)
    logger.info('saving all model output to {}'.format(model_out_filename))
    df = pd.DataFrame(output_log)
    df.to_json(model_out_filename, lines=True, orient='records')

    return


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
