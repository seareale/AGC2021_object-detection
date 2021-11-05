from pathlib import Path
import os

import yaml
import json
import numpy as np

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

LABELS = {
    'c_1':0,
    'c_2':1,
    'c_3':2,
    'c_4':3,
    'c_5':4,
    'c_6':5,
    'c_7':6
    }

CLASS_NUM = len(LABELS_NAME.keys())


def get_true_annotation(config):
    with open(config) as f:
        hyp = yaml.safe_load(f)
    path = hyp['path']

    files = []
    for p in path if isinstance(path, list) else [path]:
        p = Path('labels'.join(p.split('images')))
        if p.is_dir():
            files += Path(p).rglob('*.*')
        else:
            raise Exception(f'{p} is not directory')
    files = sorted(files)

    true_class_dict = {}
    for txt_file in files:
        filename = txt_file.name[:-4]
        true_class_dict[filename] = {}
        with open(txt_file, 'r') as f:
            bboxes = f.readlines()
        for bbox in bboxes:
            if int(bbox[0]) not in true_class_dict[filename].keys():
                true_class_dict[filename][int(bbox[0])] = 1
            else:
                true_class_dict[filename][int(bbox[0])] += 1
    
    return true_class_dict

def get_pred_annotation(results_json):
    with open(results_json) as f:
        data = json.load(f)

    pred_class_dict = {}
    for file in data['answer']:
        file_count = {}
        for cls_count in file['result']:
            file_count[LABELS[cls_count['label']]] = cls_count['count']
        pred_class_dict[file['file_name'][:-4]] = file_count

    return pred_class_dict

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

    return sum(f1score)/len(f1score), f1score

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

def show_confMat(conf_mat_list, exp_name):
    conf_list = []
    conf_norm_list = []

    for i in range(CLASS_NUM):
        conf = plot_confusion_matrix(conf_mat_list[i][0], LABELS_NAME[i], exp_name, target_names=conf_mat_list[i][1], cmap=None, normalize=False, labels=True, title='Confusion matrix')
        conf_norm = plot_confusion_matrix(conf_mat_list[i][0], LABELS_NAME[i], exp_name, target_names=conf_mat_list[i][1], cmap=None, normalize=True, labels=True, title='Confusion matrix')

        conf_list.append(conf)
        conf_norm_list.append(conf_norm)

    return conf_list, conf_norm_list

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