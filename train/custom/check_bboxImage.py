import cv2
from PIL import Image

import matplotlib.pyplot as plt
import json
import numpy as np

from pathlib import Path


LABELS = {
    'paper': 0,
    'paperpack': 1,
    'can': 2,
    'glass': 3,
    'pet': 4,
    'plastic': 5,
    'vinyl': 6,
}

LABELS_C = {
    'c_1': 0,
    'c_2': 1,
    'c_3': 2,
    'c_4': 3,
    'c_5': 4,
    'c_6': 5,
    'c_7': 6,
}

COLOR = {
    0 : (255, 0, 0),
    1 : (0, 255, 0),
    2 : (0, 0, 255),
    3 : (122, 122, 0),
    4 : (0, 122, 122),
    5 : (122, 0, 122),
    6 : (102, 34, 84),
}

LABELS_NAME = {
    0 : 'paper',
    1 : 'paperpack',
    2 : 'can',
    3 : 'glass',
    4 : 'pet',
    5 : 'plastic',
    6 : 'vinyl',
}



def load_img(path, opt='cv2', rotate=None, flip=None):
    if opt == 'cv2':
        img = cv2.imread(path)
        h,w = img.shape[:2]
        if rotate != None:
            img = cv2.rotate(img, rotate)
        if flip != None:
            if flip == 2:
                img = cv2.flip(img, 0)
                img = cv2.flip(img, 1)
            else:
                img = cv2.flip(img, flip)        
    elif opt == 'pil':
        img = Image.open(path)
        w,h = img.size

    return img, h, w

def load_txt(path):
    with open(path, 'r') as f:
        data = f.readlines()
    labels = []
    for bbox in data:
        bbox = bbox.split(' ')
        labels.append([int(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4].strip())])
    return labels

def load_json(path, h, w):
    with open(path, 'r') as f:
        json_data = json.load(f)
    json_labels = []
    # h, w = json_data['imageHeight'], json_data['imageWidth']
    for bbox in json_data['shapes']:
        try:
            l_class = LABELS[bbox['label']]
        except:
            l_class = LABELS_C[bbox['label'][:3]]
        points = np.array(bbox['points'])

        min_p = (max(np.min(points[:,0]),0), max(np.min(points[:,1]),0))
        max_p = (min(np.max(points[:,0]),w), min(np.max(points[:,1]),h))
        width = max_p[0]-min_p[0]
        height = max_p[1]-min_p[1]
        l_bbox = [l_class, (max_p[0]+min_p[0])/2/w, (max_p[1]+min_p[1])/2/h, width/w, height/h]

        json_labels.append(l_bbox)
    return json_labels

def plotimage(img, labels=None, show=True):
    plt.figure(figsize=(10,10))

    img_copy = np.array(img)
    h, w = img_copy.shape[:2]
    if labels != None:
        for bbox in labels:
            bbox = [bbox[0], (bbox[1]-bbox[3]/2)*w, (bbox[2]-bbox[4]/2)*h, bbox[3]*w, bbox[4]*h]
            bbox = [int(x) for x in bbox]
            c1, c2 = (bbox[1], bbox[2]), (bbox[1]+bbox[3],bbox[2]+bbox[4])
            # print(bbox)
            img_copy = cv2.rectangle(img_copy, c1, c2, COLOR[bbox[0]], 3)

            fs = int((h + w) * np.ceil(1 ** 0.5)  * 0.01)  # font size
            lw = round(fs / 10)
            tf = max(lw - 1, 1)  # font thickness
            text_w, text_h = cv2.getTextSize(LABELS_NAME[bbox[0]], 0, fontScale=lw / 3, thickness=tf)[0]
            c2 = c1[0] + text_w, c1[1] - text_h - 3

            img_copy = cv2.rectangle(img_copy, c1, c2, COLOR[bbox[0]], -1, cv2.LINE_AA)  # filled
            img_copy = cv2.putText(img_copy, LABELS_NAME[bbox[0]], (c1[0], c1[1] - 2), 0, lw / 3, 
                        (255,255,255), thickness=tf, lineType=cv2.LINE_AA)

    if show :
        plt.imshow(img_copy)

    return img_copy

def get_orgPath(img_path, base_path='/ssd_2/RecycleTrash/org'):
    filename = Path(img_path).name

    img_file_list = list(Path(base_path).rglob('*.jpg'))
    img_file_idx = {img.name:idx for idx,img in enumerate(img_file_list)}

    return str(img_file_list[img_file_idx[filename]])

def resize_img(image):
    resizeHeight = int(0.5 * image.shape[0]) 
    resizeWidth = int(0.5 * image.shape[1])
    img_resized = cv2.resize(image, (resizeWidth, resizeHeight), interpolation = cv2.INTER_CUBIC)

    return img_resized