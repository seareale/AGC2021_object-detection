import sys
sys.path.append('/ssd_2/haejin/yolov5')

from pathlib import Path
import json
import os
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix #, plot_confusion_matrix
from sklearn.metrics import f1_score

import itertools

import matplotlib.pyplot as plt


LABELS_NAME = {
    0 : 'paper',
    1 : 'paperpack',
    2 : 'can',
    3 : 'glass',
    4 : 'pet',
    5 : 'plastic',
    6 : 'vinyl',
}

CLASS_NUM = len(LABELS_NAME.keys())


def get_true_annotation(agc=True, old=False, dirname='AGC2021_sample'):
    if not agc:
        test_dir = f'../../../RecycleTrash/yolo/labels/test/'
        txt_list = list(Path(test_dir).glob('*.txt'))
    else :
        test_dir = f'../../../RecycleTrash/yolo/labels/{dirname}/'
        txt_list = list(Path(test_dir).glob('*.txt'))

    true_class_dict = {}
    for txt_file in txt_list:
        filename = txt_file.name[:-4]

        if old :
            true_class_dict[filename] = []
        else :
            true_class_dict[filename] = {}

        with open(txt_file, 'r') as f:
            bboxes = f.readlines()

        if old :
            for bbox in bboxes:
                true_class_dict[filename].append(int(bbox[0]))       
            true_class_dict[filename] = set(true_class_dict[filename])
        else :
            for bbox in bboxes:
                if int(bbox[0]) not in true_class_dict[filename].keys():
                    true_class_dict[filename][int(bbox[0])] = 1
                else:
                    true_class_dict[filename][int(bbox[0])] += 1

    return true_class_dict

def get_pred_annotation(pred_path, epoch=None, old=False):
    pred_json = f'../runs/val/{pred_path}/{ "best" if epoch is None else f"epoch_{str(epoch).zfill(3)}"}_predictions.json'

    pred_class_dict = {}
    with open(pred_json, 'r') as f:
        bboxes = json.load(f)

        if old :
            for bbox in bboxes:
                if bbox['image_id'] not in pred_class_dict.keys():
                    pred_class_dict[bbox['image_id']] = [bbox['category_id']]
                else:
                    pred_class_dict[bbox['image_id']].append(bbox['category_id'])
            for key in pred_class_dict.keys():
                pred_class_dict[key] = set(pred_class_dict[key])
        else :
            for bbox in bboxes:
                if bbox['image_id'] not in pred_class_dict.keys():
                    pred_class_dict[bbox['image_id']] = {}
                if bbox['category_id'] not in pred_class_dict[bbox['image_id']].keys():
                    pred_class_dict[bbox['image_id']][bbox['category_id']] = 1
                else:
                    pred_class_dict[bbox['image_id']][bbox['category_id']] += 1

    return pred_class_dict

def get_onehot(input_dict,total_label=list(range(CLASS_NUM))):
    sorted_list = []
    for key in list(input_dict.keys()):
        one_hot = [0 for x in range(CLASS_NUM)]
        for class_idx in input_dict[key]:
            one_hot[total_label.index(class_idx)] = 1
        sorted_list.append([key, one_hot])
    sorted_list = sorted(sorted_list, key=lambda k:k[0])

    return np.array([x[1] for x in sorted_list])

def get_f1score_2020(true_class_dict, pred_class_dict):
    for key in true_class_dict.keys():
        if key not in pred_class_dict.keys():
            pred_class_dict[key] = ()

    true_onehot = get_onehot(true_class_dict)
    pred_onehot = get_onehot(pred_class_dict)

    f1 = f1_score(true_onehot, pred_onehot, average='macro')

    return f1

def get_cls_confMat_2020(true_class_dict, pred_class_dict):
    for key in true_class_dict.keys():
        if key not in pred_class_dict.keys():
            pred_class_dict[key] = ()

    true_onehot = get_onehot(true_class_dict)
    pred_onehot = get_onehot(pred_class_dict)

    multi_conf = multilabel_confusion_matrix(true_onehot, pred_onehot)
    
    return multi_conf

def get_max_objNum(class_dict, label_num):
    max_num = 0

    for k0, v0 in class_dict.items():
        for k1, v1 in v0.items():
            if k1 == label_num and max_num < v1:
                max_num = v1

    return max_num

def get_cls_confMat_2021(label_num, true_class_dict, pred_class_dict):
    for key in true_class_dict.keys():
        if key not in pred_class_dict.keys():
            pred_class_dict[key] = {}

    gt_max = get_max_objNum(true_class_dict, label_num)
    pred_max = get_max_objNum(pred_class_dict, label_num)
    index_max = max(gt_max, pred_max)

    conf_mat = np.zeros((gt_max, index_max))
    target_names = (np.arange(1, pred_max+1), np.arange(1, gt_max+1))
    
    for key in true_class_dict.keys():
        if label_num not in true_class_dict[key].keys():
            continue
        gt = true_class_dict[key][label_num]

        if label_num not in pred_class_dict[key].keys():
            continue
        pred = pred_class_dict[key][label_num]

        conf_mat[gt-1][pred-1] += 1

    # gt_size, pred_size = conf_mat.shape
    # if gt_size > pred_size:
    #    conf_mat = np.concatenate((conf_mat, np.zeros((gt_size, gt_size - pred_size))), axis=1)

    return conf_mat, target_names

def get_cls_confMat_2021_zero(label_num, true_class_dict, pred_class_dict):
    for key in true_class_dict.keys():
        if key not in pred_class_dict.keys():
            pred_class_dict[key] = {}

    gt_max = get_max_objNum(true_class_dict, label_num)
    pred_max = get_max_objNum(pred_class_dict, label_num)
    index_max = max(gt_max, pred_max)

    conf_mat = np.zeros((gt_max+1, index_max+1))
    target_names = (np.arange(pred_max+1), np.arange(gt_max+1))
    
    for key in true_class_dict.keys():
        if label_num not in true_class_dict[key].keys():
            gt = 0
        else :
            gt = true_class_dict[key][label_num]
        if label_num not in pred_class_dict[key].keys():
            pred = 0
        else :
            pred = pred_class_dict[key][label_num]

        conf_mat[gt][pred] += 1
    
    # gt_size, pred_size = conf_mat.shape
    # if gt_size > pred_size:
    #     conf_mat = np.concatenate((conf_mat, np.zeros((gt_size, gt_size - pred_size))), axis=1)
    conf_mat[0][0] = 0

    return conf_mat, target_names

def get_f1score_2021(mat, include_zero=False):
    gt_max = mat.shape[0]

    precision = []
    recall = []
    f1score = []
    for i in range(gt_max):
        if include_zero and i==0:
            continue 
        tp = mat[i,i]
        fp = np.sum(mat[i,:]) - tp
        fn = np.sum(mat[:,i]) - tp

        p = tp / (tp + fp + 1e-8)
        precision.append(p)
        r = tp / (tp + fn + 1e-8)
        recall.append(r)

        f1 = 2*p*r/(p+r + 1e-8)
        f1score.append(f1)
        # print(f"{i} : {f1}")
    return sum(f1score)/len(f1score), f1score

def plot_confusion_matrix(cm, name, exp_name, target_names=None, cmap=None, normalize=False, labels=True, title='Confusion matrix'):
    f1score, _ = get_f1score_2021(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) # divide by 0
        
    fig = plt.figure(figsize=(8, 8), facecolor='w')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks_x = np.arange(len(target_names[0]))
        tick_marks_y = np.arange(len(target_names[1]))

        plt.xticks(tick_marks_x, target_names[0])
        plt.yticks(tick_marks_y, target_names[1])
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nf1score={:0.4f}; accuracy={:0.4f}; misclass={:0.4f}'.format(f1score, accuracy, misclass))
    if not os.path.exists('./AGC2021_confMat'):
        os.mkdir('./AGC2021_confMat')
    if not os.path.exists(f'./AGC2021_confMat/{exp_name}'):
        os.mkdir(f'./AGC2021_confMat/{exp_name}')
    plt.savefig(f"AGC2021_confMat/{exp_name}/{name}{'_norm' if normalize else ''}.png", dpi=300)
    plt.close(fig)
    # plt.show()

    return fig

def get_val_result(exp_name, agc=None, epoch=None, conf = 0.6, iou = 0.8, augment = False, weight=None):
    if weight != None :
        weight_list = weight
    else :
        weight_list = f'../runs/train/{exp_name}/weights/{ "best" if epoch is None else f"epoch_{str(epoch).zfill(3)}"}.pt'
    save_name = f"{exp_name.split('/')[-1]}{'_agc-all' if agc == 'all' else '_agc' if agc != None else ''}_{conf}_{iou}{'_aug' if augment else ''}"

    command = f"python ../val.py --data ../custom/data/{'agc2021_all' if agc == 'all' else 'agc2021_copy' if agc != None else 'recycle'}.yaml --weight {weight_list} --imgsz 1280 \
        --device 5 --batch-size 1 --save-json --project ../runs/val --name {save_name} --conf-thres {conf} --iou-thres {iou} \
        {'--augment' if augment else ''}"

    print(command)
    print('------------------------')
    if os.path.exists(f'../runs/val/{save_name}'):
        print('>> already exists!')
    else:
        print(os.system(command))

    return save_name

def get_detect_result(exp_name, dirname='AGC2021_sample', epoch=None, conf=0.6, iou=0.8, detect_num=1000, augment=False, weight=None):
    if weight != None :
        weight_list = weight
    else :
        weight_list = f'../runs/train/{exp_name}/weights/{ "best" if epoch is None else f"epoch_{str(epoch).zfill(3)}"}.pt'
    save_name = f"{exp_name.split('/')[-1]}_{detect_num}_{conf}_{iou}{'_aug' if augment else ''}"

    command = f"python ../detect.py --source /home/haejin/ssd_2/RecycleTrash/yolo/images/{dirname} --weight {weight_list} --imgsz 1280 \
        --conf-thres {conf} --iou-thres {iou} --max-det {detect_num} {'--augment' if augment else ''} \
        --device 5 --save-txt --save-conf --project ../runs/detect --name {save_name}"

    print(command)
    print('------------------------')
    if os.path.exists(f'../runs/detect/{save_name}'):
        print('>> already exists!')
    else:
        print(os.system(command))

    return save_name


def print_countObjNum(result_dict, label):
    count = {}
    max_num = 0
    for k0, v0 in result_dict.items():
        for k1, v1 in v0.items():
            if k1 == label :
                if v1 not in count.keys():
                    count[v1] = 1
                else :
                    count[v1] += 1
                if max_num < v1:
                    max_num = v1
    count = dict(sorted(count.items()))

    return count

def AGC2021_f1score(true_class_dict, pred_class_dict, include_zero=False):
    macro_f1 = 0
    all_f1_list = []
    conf_mat_list = []

    print(f"\n>> Macro F1 score {'- include zero' if include_zero else ''}")
    print('------------------------')
    for i in range(CLASS_NUM):
        if include_zero:
            conf_mat = get_cls_confMat_2021_zero(i, true_class_dict, pred_class_dict)
        else :
            conf_mat = get_cls_confMat_2021(i, true_class_dict, pred_class_dict)
        conf_mat_list.append(conf_mat)

        if len(conf_mat[0]) == 0 or (conf_mat[0].shape[0] == 1 and include_zero):
            continue
        else:
            result = get_f1score_2021(conf_mat[0], include_zero=include_zero)
            f1, f1_list = result
            macro_f1 += f1
            print("{:9s} : {:0.4f}".format(LABELS_NAME[i], f1))
            all_f1_list.append(result)
    print('------------------------')
    print("{:9s} : {:0.4f}".format('TOTAL', macro_f1/CLASS_NUM))

    return macro_f1/CLASS_NUM, all_f1_list, conf_mat_list

def plot_f1score(f1_list, exp_name, cmap=None, title='2021AGC F1 score'):
    f1_list = np.array([[f1 for f1, f1l in f1_list]])
    macrof1 = np.sum(f1_list)/CLASS_NUM
    target_names = list(LABELS_NAME.values())
    if 'zero' in exp_name:
        title = '2021AGC F1 score - GT,Pred 0 case'

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(10, 3), facecolor='w')
    plt.imshow(f1_list, interpolation='nearest', vmin=0, vmax=1, cmap=cmap)
    plt.title(title)

    thresh = 0.5

    for i, j in itertools.product(range(f1_list.shape[0]), range(f1_list.shape[1])):
        plt.text(j, i, "{:0.4f}".format(f1_list[i, j]),
                    horizontalalignment="center",
                    color="white" if f1_list[i, j] > thresh else "black")

    tick_marks_x = np.arange(len(target_names))
    plt.xticks(tick_marks_x, target_names)
    plt.yticks([])

    plt.tight_layout()
    plt.xlabel('Class name\n\nmacro f1 score={:0.4f}'.format(macrof1))
    if not os.path.exists('./AGC2021_confMat'):
        os.mkdir('./AGC2021_confMat')
    if not os.path.exists(f'./AGC2021_confMat/{exp_name}'):
        os.mkdir(f'./AGC2021_confMat/{exp_name}')
    plt.savefig(f"AGC2021_confMat/{exp_name}/f1score.png", dpi=300)
    plt.close(fig)

    return fig


def show_confMat(conf_mat_list, exp_name):
    conf_list = []
    conf_norm_list = []

    for i in range(CLASS_NUM):
        conf = plot_confusion_matrix(conf_mat_list[i][0], LABELS_NAME[i], exp_name, target_names=conf_mat_list[i][1], cmap=None, normalize=False, labels=True, title='Confusion matrix')
        conf_norm = plot_confusion_matrix(conf_mat_list[i][0], LABELS_NAME[i], exp_name, target_names=conf_mat_list[i][1], cmap=None, normalize=True, labels=True, title='Confusion matrix')

        conf_list.append(conf)
        conf_norm_list.append(conf_norm)

    return conf_list, conf_norm_list
