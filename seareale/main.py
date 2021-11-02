import sys
from pathlib import Path

FILE = Path(__file__).resolve()
SAVE_DIR = FILE.parents[0].as_posix()
SAVE_FILE = 'answersheet_4_03_seareale.json'
sys.path.append(SAVE_DIR)

import os
import time
import yaml
import json
import torch

from utils.datasets import LoadImages
from utils.general import non_max_suppression


def load_hyp(hyp_path):
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)

    return hyp['weights'], hyp['path'], hyp['names'], \
        hyp['imgsz'], hyp['conf'], hyp['iou'], hyp['max-det'], \
        hyp['agnostic-nms'], hyp['augment'], hyp['half'], hyp['fuse']

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    all_t1 = time.time()

    # load config and hyp
    weights, source, names, imgsz, conf_thres, iou_thres, max_det, agnostic_nms, augment, half, fuse = load_hyp('config.yaml')
    imgsz = [imgsz]*2

    # load model
    device = torch.device('cuda:0')
    ckpt = torch.load(weights[0], map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    stride = int(model.stride.max())
    if fuse:
        model.fuse()
    model.eval()
    if half:
        model.half()  # to FP16

    # load datasets
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
    # TODO
    ##################
    # TTA
    #
    #
    #

    # run once
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters()))) 

    all_t2 = time.time()
    time_load = all_t2 - all_t1

    # inference
    dict_json = {'answer':[]}
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for path, img, im0 in dataset:
        t1 = time_sync() # start time

        # Process img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync() # img load time
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=augment)[0]
        t3 = time_sync() # inference time
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, multi_label=False, max_det=max_det)
        t4 = time_sync() # NMS time
        dt[2] += t4 - t3

        # create json for results
        dict_file = {
            'file_name':f"{path.split('/')[-1]}",
            'result':[]
        }
        dict_count = {}

        for i, det in enumerate(pred):  # per image (= per one TTA)
            seen += 1

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for cls_num in det[:,5]:
                # TODO
                ##################################                
                # Remove outbound bbox
                #
                #
                #

                cls_num = int(cls_num) 
                if cls_num not in dict_count.keys():
                    dict_count[cls_num] = 1
                else :
                    dict_count[cls_num] += 1
        
        for k,v in dict_count.items():
            dict_file['result'].append({'label':names[k],'count':v})
        dict_json['answer'].append(dict_file)

        t5 = time_sync() # json time
        dt[3] += t5 - t4
        
    all_t3 = time.time()
    time_det = all_t3 - all_t2
    time_all = all_t3 - all_t1

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    t = tuple(x / seen for x in dt)  # speeds per image
    # print(f">> Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    print(f">> Speed: %.6fs pre-process, %.6fs inference, %.6fs NMS, %.6fs json, %.6fs total per image at shape {(1, 3, *imgsz)}" % (*t, sum(t)))
    print(f"          %.6fs pre-process, %.6fs inference, %.6fs NMS, %.6fs json, %.6fs total" % (*dt, sum(dt)))
    print(f">> Time : model load - %.6fs" % time_load)
    print(f"           detection - %.6fs" % time_det)
    print(f"                 all - %.6fs" % time_all)

    # Save results json
    if os.path.exists(f"{SAVE_DIR}/{SAVE_FILE}"):
        os.remove(f"{SAVE_DIR}/{SAVE_FILE}")
    with open(f"{SAVE_DIR}/{SAVE_FILE}", 'w') as f:
        json.dump(dict_json,f)
    print(f">> Results saved to {SAVE_DIR}/{SAVE_FILE}")