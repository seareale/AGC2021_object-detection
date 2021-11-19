import sys

import torch

sys.path.append("../")
import yaml
from models.yolo import Model


def weight2pretrained(ckpt_path):
    ckpt = torch.load(ckpt_path)

    ckpt["ema"] = None
    ckpt["updates"] = None
    ckpt["optimizer"] = None
    ckpt["epoch"] = -1

    torch.save(ckpt, ckpt_path)


def get_detector_init_weight(weight_path, target_path, cfg=None, ckpt=None, mode=1):
    # cfg = '../models/hub/yolov5x6.yaml'
    if (cfg == None and ckpt == None) or (cfg != None and ckpt != None):
        print(">> Wrong input!")
        return

    nc = 7
    with open("../custom/hyp/hyp.scratch.yaml") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    if ckpt != None:
        coco_ckpt = torch.load(ckpt)

    model = Model(cfg or coco_ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors"))

    train_ckpt = torch.load(weight_path)
    train_ckpt_model = train_ckpt["model"]

    # 10 = SPP
    # 11 ~ 32 = Neck
    # 33 = Detector

    if mode == 1:
        layer_num = 11
    else:
        layer_num = 33

    for i in range(layer_num, 34):
        train_ckpt_model.model[i] = model.model[i]
    train_ckpt["model"] = train_ckpt_model

    exp_name = weight_path.split("/")[-3]
    init_name = "neck-det" if mode == 1 else "det"
    base_name = "coco" if cfg == None else "rand"

    torch.save(train_ckpt, f"{target_path}/{exp_name}_{base_name}_{init_name}.pt")


# CFG_PATH = f'../models/hub/yolov5x6.yaml'
# CKPT_PATH = f"../yolov5x6.pt"
# CHECKPOINT_PATH=f"../runs/train/recycle_x6_bboxcutmix/weights/best.pt"
# TARGET_PATH = f"./weights"

# get_detector_init_weight(CHECKPOINT_PATH, TARGET_PATH, cfg=CFG_PATH)
# get_detector_init_weight(CHECKPOINT_PATH, TARGET_PATH, cfg=CFG_PATH, mode=2)

# get_detector_init_weight(CHECKPOINT_PATH, TARGET_PATH, ckpt=CKPT_PATH)
# get_detector_init_weight(CHECKPOINT_PATH, TARGET_PATH, ckpt=CKPT_PATH, mode=2)
