import sys
from pathlib import Path

FILE = Path(__file__).resolve()
SAVE_DIR = FILE.parents[0].as_posix()
SAVE_FILE = 'answersheet_4_03_seareale.json'
sys.path.append(SAVE_DIR)

import os
import time
import json
from tqdm import tqdm

import torch
from utils.datasets import LoadImages
from utils.general import *

import odach as oda

TTA_AUG_LIST = [oda.Rotate90Left(), oda.Rotate90Right(), oda.HorizontalFlip(), oda.VerticalFlip(), oda.Multiply(0.9), oda.Multiply(1.1)]


if __name__ == "__main__":
    all_t1 = time.time()


    # 1. Loading
    ######################################################################################
    # load config and hyp

    with open(f"{SAVE_DIR}/config/config.yaml") as f:
        hyp = yaml.safe_load(f)    
    imgsz = [hyp['imgsz']]*2

    # load model
    device = torch.device('cuda:0')
    ckpt = torch.load(f"{SAVE_DIR}/{hyp['weights'][0]}", map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    stride = int(model.stride.max())
    if hyp['fuse']:
        model.fuse()
    model.eval()
    if hyp['half']:
        model.half()  # to FP16

    # load datasets
    dataset = LoadImages(hyp['path'], img_size=imgsz, stride=stride, auto=False if hyp['tta'] else True)
    ######################################################################################

    # run once
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters()))) 

    # inits for TTA
    if hyp['tta']:
        TTA_AUG = [x for i, x in enumerate(TTA_AUG_LIST) if i in hyp['tta-aug']]
        TTA_SCALE = hyp['tta-scale']

        yolov5 = oda.wrap_yolov5(model, non_max_suppression)
        tta_model = oda.TTAWrapper(yolov5, TTA_AUG, TTA_SCALE)

    all_t2 = time.time()
    time_load = all_t2 - all_t1


    # 2. Prediction
    ######################################################################################
    # inference
    dict_json = {'answer':[]}
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    box_count = [0, 0, 0, 0]
    for path, img, im0 in tqdm(dataset):
        t1 = time_sync() # start time

        # Process img
        img = torch.from_numpy(img).to(device)
        img = img.half() if hyp['half'] else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync() # img load time
        dt[0] += t2 - t1

        if hyp['tta']:
            boxes, scores, labels = tta_model(img)
            pred = np.concatenate([boxes,np.array([scores]).T,np.array([labels]).T], axis=1)

            # TTA conf_thres
            pred = [pred[pred[:,4] > hyp['tta-conf']]]

            t3 = t4 = time_sync() # inference time
            dt[1] += t3 - t2
            dt[2] = dt[1]
        else:
            # Inference
            pred = model(img)[0]

            t3 = time_sync() # inference time
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, hyp['conf'], hyp['iou'], None, hyp['agnostic-nms'], multi_label=False, max_det=hyp['max-det'])
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

            # Rescale boxes from img_size to im0 size
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # check bboxes 
            h,w = im0.shape[:2]
            img_area = h*w
            bboxes = xyxy2xywh(det[:, :4])
            for idx, bbox in enumerate(bboxes):
                box_count[0] += 1

                # Remove outbound bbox
                if is_outbound(bbox, (w,h), offset=hyp['outbound']):
                    box_count[1] += 1
                    det[idx, 5] = -1

                obj_area = bbox[2]*bbox[3]
                # remove oversize bbox
                if obj_area/img_area > hyp['bbox-over']:
                    box_count[2] += 1
                    det[idx, 5] = -1

                # remove undersize bbox
                if obj_area < (h/hyp['bbox-under'] * w/hyp['bbox-under']):
                    box_count[3] += 1
                    det[idx, 5] = -1

            # count objects
            for cls_num in det[:,5]:
                cls_num = int(cls_num) 
                if cls_num not in dict_count.keys():
                    dict_count[cls_num] = 1
                else :
                    dict_count[cls_num] += 1
        
        # results to dict
        for k,v in dict_count.items():
            dict_file['result'].append({'label':hyp['names'][k],'count':str(v)})
        dict_json['answer'].append(dict_file)

        t5 = time_sync() # json time
        dt[3] += t5 - t4
        
    all_t3 = time.time()
    time_det = all_t3 - all_t2
    time_all = all_t3 - all_t1
    ######################################################################################


    # 3. Print results
    ######################################################################################
    t = tuple(x / seen for x in dt)  # speeds per image
    # print(f">> Speed : %.6fs pre-process, %.6fs inference, %.6fs NMS, %.6fs json, %.6fs total per image at shape {(1, 3, *imgsz)}" % (*t, sum(t)))
    # print(f"          %.6fs pre-process, %.6fs inference, %.6fs NMS, %.6fs json, %.6fs total" % (*dt, sum(dt)))
    print(f">> Time : model load - %.6fs" % time_load)
    print(f"           detection - %.6fs" % time_det)
    print(f"                 all - %.6fs" % time_all)
    print(f">> Results : All({box_count[0]}), Outbound({box_count[1]}), Over-size({box_count[2]}), Under-size({box_count[3]})")
    print(f"-------------------------------------------------------------------------------------")

    # Save results json
    if os.path.exists(f"{SAVE_DIR}/{SAVE_FILE}"):
        os.remove(f"{SAVE_DIR}/{SAVE_FILE}")
    with open(f"{SAVE_DIR}/{SAVE_FILE}", 'w') as f:
        json.dump(dict_json,f)
    print(f">> Results saved to {SAVE_DIR}/{SAVE_FILE}")
    ######################################################################################