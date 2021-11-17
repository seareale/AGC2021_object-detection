from utils.datasets import create_dataloader
from utils.general import colorstr
import yaml
import os


def get_data_dict(yaml_path):
    with open(yaml_path) as f:
        data_dict = yaml.safe_load(f)  # data dict

    base_path = data_dict['path']
    base_path = '../../../RecycleTrash/yolo/images/'
    train_path = [ base_path+x for x in data_dict['train']]

    return train_path

def get_hyp_dict(yaml_path):
    with open(yaml_path) as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3

    return hyp

def create_data(train_path, hyp, save_dir):
    imgsz = 1280
    batch_size = 1
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    gs = max(64, 32) 
    single_cls = False
    rect = False
    RANK = int(os.getenv('RANK', -1))
    image_weights = False
    quad = False 

    workers = 8
    cache_images = False

    return create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                hyp=hyp, augment=True, cache=cache_images, rect=rect, rank=RANK,
                                                workers=workers,
                                                image_weights=image_weights, quad=quad, prefix=colorstr('train: '),
                                                save_dir=save_dir)