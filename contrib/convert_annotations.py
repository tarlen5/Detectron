"""
author: Timothy C. Arlen
        tim.arlen@geniussports.com

date:   12 March 2018

Convert annotations to/from various types
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import glob
import os

import cv2
import pandas as pd

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

from coco_helpers import get_2017_categories

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'


def parse_voc_xml_file(filepath, image_dir):
    assert filepath.endswith(XML_EXT), "Unsupported file format"
    parser = etree.XMLParser(encoding=ENCODE_METHOD)
    xmltree = ElementTree.parse(filepath, parser=parser).getroot()
    filename = xmltree.find('filename').text.split('/')[-1]
    try:
        verified = xmltree.attrib['verified']
        if verified == 'yes':
            is_verified = True
    except KeyError:
        is_verified = False

    shapes = []
    for object_iter in xmltree.findall('object'):
        bndbox = object_iter.find("bndbox")
        label = object_iter.find('name').text

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        shapes.append({
            'filename': filename,
            'image_dir': image_dir,
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'is_verified': is_verified
        })

    return shapes


def get_ground_truth_boxes_voc(labels_files, image_dir):
    all_labels = []
    for label_file in labels_files:
        all_labels += parse_voc_xml_file(label_file, image_dir)
    df_labels = pd.DataFrame(all_labels)
    df_labels.sort_values(by='filename', inplace=True)
    df_labels.reset_index(drop=True, inplace=True)

    return df_labels

def _get_category_id(categories, label):

    if label == 'basketball':
        label = 'sports ball'

    for item in categories:
        if item['name'] == label:
            return item['id']

    return None

def convert_boxes_to_coco(df_labels, use_cat=None):

    categories = get_2017_categories()
    annotations = []
    images = []
    image_filenames = []
    categories_used = set()
    for irow, row in df_labels.iterrows():
        img_id = row['filename'].split('/')[-1].split('.')[0]
        file_name = os.path.join(row['image_dir'], row['filename'])

        if file_name not in image_filenames:
            image_filenames.append(file_name)
            img = cv2.imread(file_name)
            images.append({
                "file_name": file_name,
                "height": img.shape[0],
                "width": img.shape[1],
                "id": len(image_filenames) })
            img_id = images[-1]['id']
        else:
            img_id = None
            for item in images:
                if item['file_name'] == file_name:
                    img_id = item['id']
                    break

        assert img_id is not None, "img_id not found!"
                
        xval = row['xmin']
        yval = row['ymin']
        width = row['xmax'] - row['xmin']
        height = row['ymax'] - row['ymin']

        cat_id = _get_category_id(categories, row['label'])
        if (use_cat is not None) and (cat_id != use_cat):
            continue

        categories_used.add(cat_id)
        single_annot = {
            "area": width*height,
            "bbox": [xval, yval, width, height],
            "category_id": cat_id,
            "image_id": img_id,
            "iscrowd": 0,
            "id": irow}
        annotations.append(single_annot)

    # Only retain the categories that actually showed up in the annotations
    cats_to_save = [item for item in categories if item['id'] in categories_used]

    gt_annots = {
        'categories': cats_to_save,
        'annotations': annotations,
        'images': images}

    return gt_annots


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--im_dir',
        type=str,
        required=True,
        help='Directory to images that are annotated')
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Filename or directory where images are annotated')
    parser.add_argument(
        '--outfile',
        type=str,
        required=True,
        help='Output filename [.json]')
    parser.add_argument(
        '--cat_id',
        type=int,
        required=False,
        default=None,
        help='Use specific category. Default use all')
    parser.add_argument(
        '--input_fmt',
        type=str,
        required=False,
        default='voc',
        choices=['voc'],  # Add more here later
        help='Input type string for annotation file format')
    parser.add_argument(
        '--output_fmt',
        type=str,
        required=False,
        default='coco',
        choices=['coco'],  # Add more here later
        help='Output type string for annotation file format')
    args = parser.parse_args()

    labels_files = glob.glob(os.path.join(args.labels,'*.xml'))
    
    if args.input_fmt == "voc":
       df_labels = get_ground_truth_boxes_voc(labels_files, args.im_dir)
    else:
        raise NotImplementedError(
            'args input format: {} not implemented'.format(args.input_fmt))
    
    # Convert bbox labels to COCO format:
    boxes_coco = convert_boxes_to_coco(df_labels, use_cat=args.cat_id)

    print('writing annotations in coco format to file \n  {}'.format(args.outfile))
    with open(args.outfile, 'w') as ofile:
        json.dump(boxes_coco, ofile)
    print('Finished!')
