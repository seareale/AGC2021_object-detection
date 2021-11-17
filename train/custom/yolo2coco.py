import mmcv

import json
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import os.path as osp
import numpy as np
import imagesize
import exifread
import sys


dataset_dict = {
    '2020AGC' : ['201028_kvl', '201031_data', '201103_data', '201107_data',
        '201110_data', '201114_data', '201117_data', '201118_data', '201127_data',
        '201208_img', '201209_last'],
    'followstudy' : ['210727_easy', '210727_hard', '210809_easy', '210809_hard',
        '210901_easy', '210901_hard', '210906_easy', '210906_hard'],
    '2021AGC_test' : ['for2021AGC_1', 'for2021AGC_2', 'test_2021AGC'],
    '2021AGC' : ['211014_data', 'AGC2021_sample', 'test_2021AGC']
}

dataset_split = {
    'train' : ['211014_data'],
    'valid' : ['AGC2021_sample', 'test_2021AGC']
}

LABELS = {
    'paper': 1,
    'paperpack': 2,
    'can': 3,
    'glass': 4,
    'pet': 5,
    'plastic': 6,
    'vinyl': 7,
}

BASE_PATH='/home/haejin/ssd_2/RecycleTrash/yolo'
PREFIX='1014'

categories = []

# init categories
for idx, (k, v) in enumerate(LABELS.items()):
    # if idx % 2 == 0:
    temp = {
        "id": v,
        "name": k
    }
    categories.append(temp)

def convert_json_dict(img_idx, img_file, txt_file, obj_count):

    anno_list = []

    with open(str(txt_file), 'r') as f:
        data = f.readlines()
    # if bbox num is zero, skip 
    if len(data) == 0:
        return None

    # image dict
    # img = cv2.imread(str(img_file))
    # h,w = img.shape[:2]
    w, h = imagesize.get(str(img_file))
    with open(str(img_file), 'rb') as f:
            tags = exifread.process_file(f)
            try:
                ori = tags['Image Orientation'].values[0]
                if ori in [6,8]:
                    w, h = h, w
            except:
                pass
    img_dict = {"id" : img_idx,
        "height" : h, # img-size (height)
        "width" : w, # img-size (width)
        "file_name" : '/'.join(str(txt_file).split('/')[-2:])[:-3]+'jpg'}

    for bbox in data:
        bbox = bbox.strip().split(' ')
        class_num = int(bbox[0]) + 1
        bbox_size = (int(float(bbox[3])*w), int(float(bbox[4])*h))
        bbox_center = (float(bbox[1])*w, float(bbox[2])*h)
        pt1 = (int(bbox_center[0]-bbox_size[0]/2), int(bbox_center[0]+bbox_size[0]/2))
        pt2 = (int(bbox_center[1]-bbox_size[1]/2), int(bbox_center[1]+bbox_size[1]/2))
        
        anno_dict = {"id" : obj_count,
            "image_id" : img_idx,
            "category_id" : class_num,
            "segmentation" : [[pt1[0],pt1[1],pt2[0],pt1[1],pt2[0],pt2[1],pt1[0],pt2[1]]],
            "area" : bbox_size[0] * bbox_size[1],
            "bbox" : [pt1[0], pt1[1], bbox_size[0], bbox_size[1]],
            "iscrowd" : 0}

        anno_list.append(anno_dict)

        obj_count += 1

    return img_dict, anno_list, obj_count

def get_coco_json(base_path, prefix):
    images = {'train' : [],
        'valid' : []}
    annotations = {'train' : [],
        'valid' : []}

    img_list = sorted(list(Path(f"{base_path}/images").rglob("*.jpg")))
    txt_list = sorted(list(Path(f"{base_path}/labels").rglob("*.txt")))

    if not osp.exists(f"{base_path}/annotations"):
        os.mkdir(f"{base_path}/annotations")

    print(f'Convert YOLO to COCO!!! - ({prefix})')

    img_count, obj_count = 0, 0
    for img_path, txt_path in tqdm(zip(img_list, txt_list)):

        #################################################################################
        if str(img_path).split('/')[-2] not in dataset_dict['2021AGC']:
            continue
        #################################################################################

        results = convert_json_dict(img_count, img_path, txt_path, obj_count)

        if results is None:
            continue
        else :
            img_dict, anno_list, obj_count = results

            #################################################################################
            if str(img_path).split('/')[-2] in dataset_split['train']:
                images['train'].append(img_dict)
                annotations['train'] += anno_list
            elif str(img_path).split('/')[-2] in dataset_split['valid']:
                images['valid'].append(img_dict)
                annotations['valid'] += anno_list
            #################################################################################

            # images.append(img_dict)
            # annotations += anno_list

            img_count += 1
        
    coco_format_json = {
        'train' : dict(
            images=images['train'],
            annotations=annotations['train'],
            categories=categories),
        'valid' : dict(
            images=images['valid'],
            annotations=annotations['valid'],
            categories=categories),
        'all' : dict(
            images=images['train']+images['valid'],
            annotations=annotations['train']+annotations['valid'],
            categories=categories)
    }

    if osp.exists(f"{base_path}/annotations/{prefix}_all.json"):
        os.remove(f"{base_path}/annotations/{prefix}_all.json")
    if osp.exists(f"{base_path}/annotations/train.json"):
        os.remove(f"{base_path}/annotations/train.json")
    if osp.exists(f"{base_path}/annotations/valid.json"):
        os.remove(f"{base_path}/annotations/valid.json")

    mmcv.dump(coco_format_json['all'], f"{base_path}/annotations/{prefix}_all.json")
    mmcv.dump(coco_format_json['train'], f"{base_path}/annotations/train.json")
    mmcv.dump(coco_format_json['valid'], f"{base_path}/annotations/valid.json")



def main(start_img_id, start_obj_id, base_path, prefix, target_path):
    images = []
    annotations = []

    img_list = sorted(list(Path(f"{target_path}/images").rglob("*.jpg")))
    txt_list = sorted(list(Path(f"{base_path}/labels").rglob("*.txt")))

    if not osp.exists(f"{target_path}/annotations"):
        os.mkdir(f"{target_path}/annotations")

    print(f'Convert YOLO to COCO!!! - ({prefix})')

    img_count = start_img_id
    obj_count = start_obj_id
    for img_path, txt_path in tqdm(zip(img_list, txt_list)):


        if str(img_path).split('/')[-2] not in dataset_dict['2021AGC']:
            continue


        results = convert_json_dict(img_count, img_path, txt_path, obj_count)

        if results is None:
            continue
        else :
            img_dict, anno_list, obj_count = results

            images.append(img_dict)
            annotations += anno_list

            img_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)

    if osp.exists(f"{target_path}/annotations/{prefix}_all.json"):
        os.remove(f"{target_path}/annotations/{prefix}_all.json")

    mmcv.dump(coco_format_json, f"{target_path}/annotations/{prefix}_all.json")

if __name__ == "__main__":
    # img_id = int(sys.argv[1])
    # obj_id = int(sys.argv[2])

    get_coco_json(BASE_PATH, PREFIX)

    # main(img_id, obj_id, BASE_PATH, PREFIX, TARGET_PATH)