import json
from pathlib import Path

import imagesize


def is_outbound(points, image_size, offset=10):
    w, h = image_size
    boundary = [(10, 10), (w - 10, 10), (w - 10, h - 10), (10, h - 10)]
    coord_1 = int(points[0] - points[2] / 2), int(points[1] - points[3] / 2)
    coord_2 = int(points[0] + points[2] / 2), int(points[1] - points[3] / 2)
    coord_3 = int(points[0] + points[2] / 2), int(points[1] + points[3] / 2)
    coord_4 = int(points[0] - points[2] / 2), int(points[1] + points[3] / 2)
    points = [coord_1, coord_2, coord_3, coord_4]

    for idx, ((x0, y0), (x1, y1)) in enumerate(zip(points, boundary)):
        if idx == 0 and (x0 < x1 or y0 < y1):
            return True
        if idx == 1 and (x0 > x1 or y0 < y1):
            return True
        if idx == 2 and (x0 > x1 or y0 > y1):
            return True
        if idx == 3 and (x0 < x1 or y0 > y1):
            return True

    return False


def is_outbound(points, image_size, offset=10):
    w, h = image_size
    boundary = [(10, 10), (w - 10, 10), (w - 10, h - 10), (10, h - 10)]
    coord_1 = int(points[0] - points[2] / 2), int(points[1] - points[3] / 2)
    coord_2 = int(points[0] + points[2] / 2), int(points[1] - points[3] / 2)
    coord_3 = int(points[0] + points[2] / 2), int(points[1] + points[3] / 2)
    coord_4 = int(points[0] - points[2] / 2), int(points[1] + points[3] / 2)
    points = [coord_1, coord_2, coord_3, coord_4]

    for idx, ((x0, y0), (x1, y1)) in enumerate(zip(points, boundary)):
        if idx == 0 and (x0 < x1 or y0 < y1):
            return True
        if idx == 1 and (x0 > x1 or y0 < y1):
            return True
        if idx == 2 and (x0 > x1 or y0 > y1):
            return True
        if idx == 3 and (x0 < x1 or y0 > y1):
            return True

    return False


def get_unoutbbox_txt(txt_path, base_path):
    txt_list = list(Path(txt_path).glob("*.txt"))

    results = []

    count = 0

    for txt_file in txt_list:
        img_path = f"{base_path}/{txt_file.name[:-4]}.jpg"
        image_size = imagesize.get(img_path)
        w, h = image_size

        with open(txt_file, "r") as f:
            data = f.readlines()

        count += len(data)

        for bbox in data:
            bbox = bbox.split(" ")
            labels = [
                int(bbox[0]),
                float(bbox[1]) * w,
                float(bbox[2]) * h,
                float(bbox[3]) * w,
                float(bbox[4].strip()) * h,
            ]

            if is_outbound(labels[1:], image_size):
                print(img_path)
                print(labels[1:])
                print("-------------------------------------------------")
            else:
                dict_txt = {
                    "image_id": txt_file.name[:-4],
                    "category_id": labels[0],
                    "bbox": labels[1:],
                }

                results.append(dict_txt)

    print(f">> object num : {count}")

    return results


#################

# # detect.py 실행
# exp_name = "agc2021_x6_1027_a1026_90-rot_rand-rot-5_backbone7"
# # 'agc2021_x6_1022_1014_90-rot_rand-rot-5'
# detect_num = 30
# conf = 0.6
# iou = 0.8
# save_name = get_detect_result(
#     exp_name, epoch=None, conf=conf, iou=iou, detect_num=detect_num, augment=False, weight=None
# )
# save_name = get_val_result(
#     exp_name, agc="one", epoch=None, conf=conf, iou=iou, augment=False, weight=None
# )


# exp_name = "agc2021_x6_1026_a1025_90-rot_rand-rot-5"
# # json_file = '../runs/val/agc2021_x6_1026_a1025_90-rot_rand-rot-5_agc_0.6_0.8/best_predictions.json'
# json_file = f"../runs/val/{exp_name}_agc_0.6_0.8/best_predictions.json"
# txt_path = f"../runs/detect/{exp_name}_{detect_num}_0.6_0.8/labels"
# base_path = "../../../RecycleTrash/yolo/images/AGC2021_sample"

# results = get_unoutbbox_json(json_file, base_path)
# results = get_unoutbbox_txt(txt_path, base_path)
