import mmcv

import json
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import os.path as osp
import numpy as np
import imagesize
import exifread
import sys


dataset_dict = {
    'train' : ['211014_data'],
    'valid' : ['AGC2021_sample', 'test_2021AGC']
}