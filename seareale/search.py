import sys
from os import path
from pathlib import Path

FILE = Path(__file__).resolve()
SAVE_DIR = FILE.parents[0].as_posix()
SAVE_OUTPUT = "output_results.json"
SAVE_F1 = "f1_results.json"
sys.path.append(FILE.parents[1].as_posix())

import json
import os
import warnings
from collections import defaultdict
from itertools import product

import torch
import yaml
from test_tools.f1score import *
from tqdm import tqdm

import odach as oda
from utils.datasets import LoadImages
from utils.general import *

TTA_AUG_LIST = [
    oda.Rotate90Left(),
    oda.Rotate90Right(),
    oda.HorizontalFlip(),
    oda.VerticalFlip(),
    oda.RandColorJitter(),
    oda.TorchBlur(),
    oda.TorchMedianBlur(),
]

if __name__ == "__main__":
    warnings.filterwarnings(action="ignore")
    ######################################################################################
    # 1. Loading
    ######################################################################################
    # load config and hyp

    with open(f"{SAVE_DIR}/config/config.yaml") as f:
        hyp = yaml.safe_load(f)
    imgsz = [hyp["imgsz"]] * 2
    # inits for TTA
    TTA_SCALE = hyp["tta-scale"]

    ######################################################################################
    # 2. Load JSON for each Augmentation
    ######################################################################################
    if "search_result" in hyp.keys() and os.path.exists(hyp["search_result"]):
        with open(hyp["search_result"]) as f:
            total_results = json.load(f)

    ######################################################################################
    # 2-1. Make JSON for each Augmentation
    ######################################################################################
    else:
        # load model
        device = torch.device("cuda:0")
        ckpt = torch.load(f"{SAVE_DIR}/{hyp['weights'][0]}", map_location=device)
        model = ckpt["ema" if ckpt.get("ema") else "model"].float()
        stride = int(model.stride.max())
        if hyp["fuse"]:
            model.fuse()
        model.eval()
        if hyp["half"]:
            model.half()  # to FP16
        # load datasets
        dataset = LoadImages(
            hyp["path"], img_size=imgsz, stride=stride, auto=False  # if hyp["tta"] else True
        )
        loader = torch.utils.data.DataLoader
        dataloader = loader(dataset, batch_size=hyp["batchsz"], num_workers=hyp["numworker"])

        # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
        yolov5 = oda.wrap_yolov5(model, non_max_suppression)

        # Get all combinations of Auglist
        combinations = list(product(*list([i, None] for i in TTA_AUG_LIST)))
        print(
            f"Total combinations: scale {len(TTA_SCALE)} x comb {len(combinations)} = {len(TTA_SCALE)*len(combinations)}"
        )
        total_results = defaultdict(list)
        for scale_idx, s in enumerate(TTA_SCALE):
            for comb_idx, tta_combination in enumerate(combinations, 1):
                torch.cuda.empty_cache()
                # Make oda_aug
                oda_aug = oda.TTACompose(
                    [oda.MultiScale(s)]
                    + [tta_transform for tta_transform in tta_combination if tta_transform]
                )
                print(f"{scale_idx*len(combinations) + comb_idx} tta_combination: {oda_aug}")
                # Inference using oda_aug
                for path, batch, batch_orig in tqdm(dataloader):
                    # Preprocess batch
                    batch = torch.from_numpy(np.asarray(batch)).to(device)
                    batch = batch / 255.0  # 0 - 255 to 0.0 - 1.0
                    batch = oda_aug.batch_augment(batch).half()
                    with torch.no_grad():
                        results = yolov5(batch)
                        for idx, (p, orig_size, result) in enumerate(
                            zip(path, batch_orig, results)
                        ):
                            dict_file = {
                                "file_name": f"{p.split('/')[-1]}",
                                "input_size": batch.shape[-2:],
                                "orig_size": orig_size[:2].tolist(),
                                "result": {},
                            }
                            boxes = result["boxes"].cpu().numpy()
                            boxes = oda_aug.deaugment_boxes(boxes)

                            thresh = 0.01
                            ind = result["scores"].cpu().numpy() > thresh

                            dict_file["result"]["boxes"] = boxes[ind].tolist()
                            dict_file["result"]["scores"] = (
                                result["scores"].cpu().numpy()[ind].tolist()
                            )
                            dict_file["result"]["labels"] = (
                                result["labels"].cpu().numpy()[ind].tolist()
                            )
                            total_results[str(oda_aug)].append(dict_file)
                # Save results json
                if os.path.exists(f"{SAVE_DIR}/{SAVE_OUTPUT}"):
                    os.remove(f"{SAVE_DIR}/{SAVE_OUTPUT}")
                with open(f"{SAVE_DIR}/{SAVE_OUTPUT}", "w") as f:
                    json.dump(total_results, f, indent=4)
                print(f">> Results saved to {SAVE_DIR}/{SAVE_OUTPUT}")

    ######################################################################################
    # 3. Get optimal combination using JSON
    ######################################################################################
    if os.path.exists(f"{SAVE_DIR}/{SAVE_F1}"):
        print("Load previous f1 results")
        with open(f"{SAVE_DIR}/{SAVE_F1}") as f:
            f1_results = json.load(f)
    else:
        print("Create new f1 results")
        f1_results = {}

    scale_transforms = []
    for scale_list in product(*list([f"MultiScale({scl})", None] for scl in TTA_SCALE)):
        scale_transforms.append([scl_tf for scl_tf in scale_list if scl_tf is not None])
    scale_transform = scale_transforms[:-1]
    oda_transforms = product(*list([str(i), None] for i in TTA_AUG_LIST))

    nms = oda.nms_func(skip_box_thr=0.5)

    num_of_data = len(total_results[next(iter(total_results.keys()))])
    all_combinations = list(product(scale_transform, oda_transforms))
    print(f"Get performance of {len(all_combinations)} comb")
    for s_list, oda_list in tqdm(all_combinations):
        comb_name = str(s_list) + "x" + str([aug for aug in oda_list if aug is not None])
        if comb_name in f1_results.keys():
            continue
        comb_names = [None for _ in range(num_of_data)]
        comb_input_sizes = [None for _ in range(num_of_data)]
        comb_orig_sizes = [None for _ in range(num_of_data)]
        comb_bboxes = [[] for _ in range(num_of_data)]
        comb_scores = [[] for _ in range(num_of_data)]
        comb_labels = [[] for _ in range(num_of_data)]
        for s in s_list:
            for comb in product(*list([i, None] for i in oda_list)):
                augs_name = str([s] + [aug for aug in comb if aug is not None]).replace("'", "")
                for idx, result_per_img in enumerate(total_results[augs_name]):
                    comb_names[idx] = result_per_img["file_name"]
                    comb_input_sizes[idx] = result_per_img["input_size"]
                    comb_orig_sizes[idx] = result_per_img["orig_size"]
                    comb_bboxes[idx].append(result_per_img["result"]["boxes"])
                    comb_scores[idx].append(result_per_img["result"]["scores"])
                    comb_labels[idx].append(result_per_img["result"]["labels"])

        # NMS and post process for each image's result
        answers = {"answer": []}
        for idx, (file_name, input_size, orig_size, bboxes, scores, labels) in enumerate(
            zip(
                comb_names,
                comb_input_sizes,
                comb_orig_sizes,
                comb_bboxes,
                comb_scores,
                comb_labels,
            )
        ):
            answer = {"file_name": file_name, "result": []}
            # NMS
            comb_bboxes[idx], comb_scores[idx], comb_labels[idx] = nms(bboxes, scores, labels)
            pred = np.concatenate(
                [comb_bboxes[idx], np.array([comb_scores[idx]]).T, np.array([comb_labels[idx]]).T],
                axis=1,
            )
            # NMS thresholding
            pred = pred[pred[:, 4] > 0.4]
            # Post process
            pred[:, :4] = scale_coords(input_size, pred[:, :4], orig_size).round()
            h, w = orig_size
            img_area = h * w
            bboxes = xyxy2xywh(pred[:, :4])
            for box_idx, bbox in enumerate(bboxes):
                if is_outbound(bbox, (w, h), offset=hyp["outbound"]):
                    pred[box_idx, 5] = -1

                obj_area = bbox[2] * bbox[3]
                # remove oversize bbox
                if obj_area / img_area > hyp["bbox-over"]:
                    pred[box_idx, 5] = -1

                # remove undersize bbox
                if obj_area < (h / hyp["bbox-under"] * w / hyp["bbox-under"]):
                    pred[box_idx, 5] = -1

            # count objects
            dict_count = {}
            for cls_num in pred[:, 5]:
                cls_num = int(cls_num)
                if cls_num not in dict_count.keys():
                    dict_count[cls_num] = 1
                else:
                    dict_count[cls_num] += 1

            # results to dict
            num_list = list(range(7))
            for k in dict_count.keys():
                if k == -1:
                    continue
                num_list.remove(k)
            for k in num_list:
                dict_count[k] = 0

            # create answer
            for k, v in dict_count.items():
                if k == -1:
                    continue
                answer["result"].append({"label": hyp["names"][k], "count": str(v)})
            answers["answer"].append(answer)

        # Calculate f1 score of each combination
        true_dict = get_true_annotation(f"{SAVE_DIR}/config/config.yaml", one_path=None)
        pred_dict = get_pred_annotation(None, data=answers)
        f1, *_ = AGC2021_f1score(true_dict, pred_dict, print_result=False)

        f1_results[comb_name] = f1
        # sort f1_results using value
        f1_results = dict(sorted(f1_results.items(), key=lambda x: x[1], reverse=True))
        # Save results json
        if os.path.exists(f"{SAVE_DIR}/{SAVE_F1}"):
            os.remove(f"{SAVE_DIR}/{SAVE_F1}")
        with open(f"{SAVE_DIR}/{SAVE_F1}", "w") as f:
            json.dump(f1_results, f, indent=4)

    print(f">> Results saved to {SAVE_DIR}/{SAVE_F1}")
