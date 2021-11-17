import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm

import matplotlib
import pandas as pd
import seaborn as sn
from PIL import Image, ImageDraw
from utils.general import xywh2xyxy

import os
import cv2


CLASSES = ['c_1' ,'c_2' ,'c_3' ,'c_4' ,'c_5' ,'c_6' ,'c_7']

base_path = '../../../RecycleTrash/yolo/labels'

alldata = {
    'train_list' : ['201028_kvl',
        '201031_data', 
        '201103_data', '201107_data', '201110_data', '201114_data', '201117_data', '201118_data', '201127_data',
        '201208_img', '201209_last',
        '210727_easy', '210727_hard',
        '210809_easy', '210809_hard',
        '210901_easy', '210901_hard',
        '210906_easy', '210906_hard',
        'for2021AGC_1', 'for2021AGC_2'
        ],
    'valid_list':['test'],
    'test_list':['test']
}


agc2020 = {
    'train_list' : ['201028_kvl',
        '201031_data', 
        '201103_data', '201107_data', '201110_data', '201114_data', '201117_data', '201118_data', '201127_data',
        '201208_img', '201209_last'
        ],
    'valid_list':['test'],
    'test_list':['test']
}

followstudy = {
    'train_list' : [
        '210727_easy', '210727_hard',
        '210809_easy', '210809_hard',
        '210901_easy', '210901_hard',
        '210906_easy', '210906_hard'
        ],
    'valid_list':['test'],
    'test_list':['test']
}

all2020 = {
    'train_list' : ['201028_kvl',
        '201031_data', 
        '201103_data', '201107_data', '201110_data', '201114_data', '201117_data', '201118_data', '201127_data',
        '201208_img', '201209_last',
        '210727_easy', '210727_hard',
        '210809_easy', '210809_hard',
        '210901_easy', '210901_hard',
        '210906_easy', '210906_hard'
        ],
    'valid_list':['test'],
    'test_list':['test']
}

agc2021 = {
    'train_list' : ['211014_data_re', '211021_data', '211022_data', '211025_data',
        '211026_data'],
        #'211027_data', '211028_data', '211029_data', '211030_data', 
        # '211031_data', '211101_data', '211102_data', '211103_data', '211104_data'],
    'valid_list':['test_2020AGC_452', 'test_2021AGC_30', 'test_211026_405', 'test_211103_400'],
    'test_list':['test_2020AGC_452', 'test_2021AGC_30', 'test_211026_405', 'test_211103_400']
}


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

##################################################################
def plot_class_by_dataset(base_path):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)
    class_ticks = np.arange(7)

    plt.figure(figsize=(20,8))

    for idx, dirname in enumerate(dir_list):
        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
        class_count = np.zeros(7).astype(int)
        for txt in txt_list:
            with open(txt) as f:
                ann = f.readlines()
                for bbox in ann:
                    class_count[int(bbox[0])] += 1
        # print(('%12s'+' : ['+'%6d'*7+']') % (dirname, *class_count))
        class_ticks 
        plt.bar(class_ticks+idx*10, class_count)
    plt.xticks([(i+0.3)*10 for i in range(len(dir_list))], dir_list, fontsize=15, rotation=30)
    plt.show()

def count_class_by_dataset(base_path):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)

    all_count = np.zeros(7).astype(int)

    print(('%12s'+'    '+'%6s'*7) % ('', *CLASSES))
    for dirname in dir_list:
        class_count = np.zeros(7).astype(int)

        if dirname == 'test':
            continue

        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))

        for txt in txt_list:
            with open(txt) as f:
                ann = f.readlines()
                for bbox in ann:
                    class_count[int(bbox[0])] += 1
                    all_count[int(bbox[0])] += 1

        print( ('%12s'+' : ['+'%6d'*7+']') % ( dirname, *class_count))
    print('-'*(12+6*7+5))
    print( ('%12s'+' : ['+'%6d'*7+']') % ( 'all', *all_count),f"-> SUM : {sum(all_count)}", '/ MAX :', f'{max(all_count)}({round(max(all_count), -3)})')
    print('-'*(12+6*7+5))

    temp = [ max(round(max(all_count), -3) - x,0) for x in all_count]

    print( ('%12s'+' : ['+'%6d'*7+']') % ( 'we need', *temp), '-> SUM :', sum(temp))

def count_object_by_dataset(base_path):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)
    all_count = np.zeros(20).astype(int)
    test = np.zeros(20).astype(int)

    print( ('%12s'+'   ['+'%6d'*20+']') % ( '', *np.arange(20)+1))
    print('-'*(12+6*20+5))

    for dirname in dir_list:
        obj_count = np.zeros(20).astype(int)
        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
        for txt in txt_list:
            with open(txt) as f:
                num_obj = len(f.readlines())
                if 'test' in str(txt):
                    test[num_obj - 1] += 1
                obj_count[num_obj - 1] += 1
                all_count[num_obj - 1] += 1
        print( ('%12s'+' : ['+'%6d'*20+']') % ( dirname, *obj_count))

    print('-'*(12+6*20+5))
    print( ('%12s'+' : ['+'%6d'*20+']') % ( 'all', *all_count), '-> MAX :', f'{max(all_count)}({round(max(all_count), -3)})')
    print('-'*(12+6*20+5))

    # test *= 12
    test *= 55

    # test 개수에 비례
    print( ('%12s'+' : ['+'%6d'*20+']') % ( 'need # imgs', *test), '-> SUM :', sum(test))
    temp = [ x*(i+1) for i,x in enumerate(test)]
    print( ('%12s'+' : ['+'%6d'*20+']') % ( '# class', *temp), '-> SUM :', sum(temp))

##################################################################
def plot_class_by_tvt(base_path, dir_list, train_out=False):
    class_ticks = np.arange(7)

    plt.figure(figsize=(10,3))

    for idx, (k, v) in enumerate(dir_list.items()):
        if train_out and idx == 0 :
            continue
        class_count = np.zeros(7).astype(int)
        for dirname in v:
            txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
            for txt in txt_list:
                with open(txt) as f:
                    ann = f.readlines()
                    for bbox in ann:
                        class_count[int(bbox[0])] += 1
        # print(('%12s'+' : ['+'%6d'*7+']') % (k, *class_count))
        plt.bar(class_ticks+(idx-1 if train_out else idx)*10, class_count)
    
    displayNum = len(dir_list)-1 if train_out else len(dir_list)
    displayLabel = list(dir_list.keys())
    
    if train_out:
        displayLabel.remove('train_list')

    plt.xticks([(i+0.3)*10 for i in range(displayNum)], displayLabel, fontsize=15)

def count_num_by_tvt(base_path, dir_list, train_out=False):
    print(f"이미지 개수")
    all_count = 0

    for idx, (k, v) in enumerate(dir_list.items()):
        if train_out and idx == 0 :
            continue
        num_count = 0
        for dirname in v:
            txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
            num_count += len(txt_list)
            all_count += len(txt_list)
        print( ('%12s'+' : ['+'%6d'*1+']') % ( k, num_count))
    print('-'*(12+6*1+5))
    print( ('%12s'+' : ['+'%6d'*1+']\n') % ( 'all', all_count))

def count_class_by_tvt(base_path, dir_list, train_out=False):
    print(f"클래스 별 object 개수")
    all_count = np.zeros(7).astype(int)

    print(('%12s'+'    '+'%9s'*7) % ('', *CLASSES))
    for idx, (k, v) in enumerate(dir_list.items()):
        if train_out and idx == 0 :
            continue
        class_count = np.zeros(7).astype(int)
        for dirname in v:
            txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
            for txt in txt_list:
                with open(txt) as f:
                    ann = f.readlines()
                    for bbox in ann:
                        class_count[int(bbox[0])] += 1
                        all_count[int(bbox[0])] += 1
        print( ('%12s'+' : ['+'%9d'*7+'] -> sum : '+'%9d') % ( k, *class_count, class_count.sum()))
    print('-'*(12+9*7+5))
    print( ('%12s'+' : ['+'%9d'*7+'] -> sum : '+'%9d\n') % ( 'all', *all_count, all_count.sum()))

def count_object_by_tvt(base_path, dir_list, train_out=False):
    print(f"object 개수 별 이미지 개수")
    all_count = np.zeros(20).astype(int)

    print( ('%12s'+'   ['+'%6d'*20+']') % ( '', *np.arange(20)+1))
    print('-'*(12+6*20+5))
    for idx, (k, v) in enumerate(dir_list.items()):
        if train_out and idx == 0 :
            continue
        obj_count = np.zeros(20).astype(int)
        for dirname in v:
            txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))
            for txt in txt_list:
                with open(txt) as f:
                    num_obj = len(f.readlines())
                    obj_count[num_obj - 1] += 1
                    all_count[num_obj - 1] += 1
        print(('%12s'+' : ['+'%6d'*20+'] -> sum : '+'%8d') % (k, *obj_count, obj_count.sum()))
    print('-'*(12+6*20+5))
    print( ('%12s'+' : ['+'%6d'*20+'] -> sum : '+'%8d') % ( 'all', *all_count, all_count.sum()))
    print('-'*(12+6*20+5))

##########################################################################
def plot_labels_by_dataset(base_path, names, save_dir=Path('img/labels')):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)

    for dirname in dir_list:
        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))

        labels = []
        for txt in txt_list:
            with open(txt) as f:
                while True:
                    ann = f.readline()
                    if not ann: break
                    ann = ann.strip().split(' ')
                    labels.append([int(ann[0]), float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4])])
        labels = np.array(labels)

        c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
        nc = int(c.max() + 1)  # number of classes
        x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

        # seaborn correlogram
        sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
        plt.savefig(save_dir / f'{dirname}_labels_correlogram.jpg', dpi=200)
        plt.close()

        # matplotlib labels
        matplotlib.use('svg')  # faster
        ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
        y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
        # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
        ax[0].set_ylabel('instances')
        if 0 < len(names) < 30:
            ax[0].set_xticks(range(len(names)))
            ax[0].set_xticklabels(names, rotation=90, fontsize=10)
        else:
            ax[0].set_xlabel('classes')
        sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
        sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

        # rectangles
        labels[:, 1:3] = 0.5  # center
        labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
        img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
        for cls, *box in labels[:1000]:
            ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
        ax[1].imshow(img)
        ax[1].axis('off')

        for a in [0, 1, 2, 3]:
            for s in ['top', 'right', 'left', 'bottom']:
                ax[a].spines[s].set_visible(False)

        plt.savefig(save_dir / f'{dirname}_labels.jpg', dpi=200)
        matplotlib.use('Agg')
        plt.close()

def min_max_object(base_path):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)

    max_width = 0
    max_height = 0

    for dirname in dir_list:
        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))

        labels = []
        for idx, txt in enumerate(txt_list):
            with open(txt) as f:
                while True:
                    ann = f.readline()
                    if not ann: break
                    ann = ann.strip().split(' ')
                    labels.append([int(ann[0]), float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4]), idx])
        labels = np.array(labels)

        c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
        x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height', 'file_id'])

        print(f"-------------------------------------------------------")
        print(f"{dirname}")

        # area 기준
        max_width_idx = x['width'].idxmax()
        max_height_idx = x['height'].idxmax()

        img = cv2.imread('/images/'.join(str(txt_list[int(x.loc[max_width_idx].file_id)]).rsplit('/labels/', 1))[:-3]+'jpg')
        _, width, _ = img.shape
        width = int(x.loc[max_width_idx].width * width)

        if width > max_width:
            max_width = width

        img = cv2.imread('/images/'.join(str(txt_list[int(x.loc[max_height_idx].file_id)]).rsplit('/labels/', 1))[:-3]+'jpg')
        height, _, _ = img.shape
        height = int(x.loc[max_height_idx].height * height)

        if height > max_height:
            max_height = height

        print(f"max height, width : {height} x {width}")

        min_idx = (x['width']*x['height']).idxmin()
        max_idx = (x['width']*x['height']).idxmax()

        img = cv2.imread('/images/'.join(str(txt_list[int(x.loc[min_idx].file_id)]).rsplit('/labels/', 1))[:-3]+'jpg')
        height, width, _ = img.shape
        height = int(x.loc[min_idx].height * height)
        width = int(x.loc[min_idx].width * width)

        print(f"min object : {txt_list[int(x.loc[min_idx].file_id)].name} ({height} x {width})")

        img = cv2.imread('/images/'.join(str(txt_list[int(x.loc[max_idx].file_id)]).rsplit('/labels/', 1))[:-3]+'jpg')
        height, width, _ = img.shape
        height = int(x.loc[max_idx].height * height)
        width = int(x.loc[max_idx].width * width)

        print(f"max object : {txt_list[int(x.loc[max_idx].file_id)].name} ({height} x {width})")

    print('---------------------')
    print(f"max height, width : {max_height} x {max_width}")

def count_image_num(base_path, tvt_list):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)

    all_count = 0
    train_count = 0
    valid_count = 0
    test_count = 0

    for dirname in dir_list:
        num_file = len(list(Path(f'{base_path}/{dirname}').glob('*.txt')))
        print(f"{dirname:15s}:{num_file:7d}")
        all_count += num_file
        if dirname in tvt_list['train_list']:
            train_count += num_file
        elif dirname in tvt_list['valid_list']:
            valid_count += num_file
        elif dirname in tvt_list['test_list']:
            test_count += num_file
    
    print(f"{'-'*(15+3+7)}")
    print(f"{'train':15s}:{train_count:7d}")
    print(f"{'valid':15s}:{valid_count:7d}")
    print(f"{'test':15s}:{test_count:7d}")
    print(f"{'-'*(15+3+7)}")
    print(f"{'all':15s}:{all_count:7d}")

def check_distance_center(base_path):
    dir_list = sorted([x.name for x in list(Path(base_path).glob('*')) if x.is_dir()], key=lambda y: len( list(Path(f'{base_path}/{y}').glob('*'))), reverse=True)

    distance_x = []
    distance_y = []
    for dirname in dir_list:
        txt_list = list(Path(f'{base_path}/{dirname}').glob('*.txt'))

        for txt in tqdm(txt_list):
            labels = []
            try:
                img = cv2.imread('/images/'.join(str(txt).rsplit('/labels/', 1))[:-3]+'jpg')
            except:
                print(txt)
                continue
            height, width, _ = img.shape
            
            with open(txt) as f:
                while True:
                    ann = f.readline()
                    if not ann: break
                    ann = ann.strip().split(' ')
                    labels.append([float(ann[1])*width, float(ann[2])*height])
            
            labels = np.array(labels)

            for i, (x1,y1) in enumerate(labels):
                if i == len(labels)-1:
                    break
                for (x2,y2) in labels[i+1:]:
                    distance_x.append(max(x1,x2) - min(x1,x2))
                    distance_y.append(max(y1,y2) - min(y1,y2))
        print(f"[{dirname}] - end")

    print(np.mean(distance_x), np.mean(distance_y))

    return distance_x, distance_y


# count_image_num(base_path, new)
# min_max_object(base_path)

# count_class_by_dataset(base_path)
# count_objectNum_by_dataset(base_path)
# plot_class_by_dataset(base_path)

# plot_labels_by_dataset(base_path, CLASSES) # 각 dataset별로 label 분석 

def get_tvt_info(base_path, data, train_out=False):
    plot_class_by_tvt(base_path, data, train_out=train_out)
    count_num_by_tvt(base_path, data)
    count_class_by_tvt(base_path, data)
    count_object_by_tvt(base_path, data)