# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

#######################################
# TODO: ImbalancedDatasetSampler, bbox-cutmix 추가
from custom.imbalanced import ImbalancedDatasetSampler
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    check_dataset,
    check_requirements,
    check_yaml,
    clean_str,
    segments2boxes,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

#######################################

# Parameters
HELP_URL = "https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"
IMG_FORMATS = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # acceptable image suffixes
VID_FORMATS = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
]  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    save_dir=None,
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache

    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            dist=rank,
            save_dir=save_dir,
        )
        # TODO: dataset mapping

        # dataset.share_memory()
    # import torch.distributed as dist
    # dist.barrier()

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers

    #######################################
    # TODO: ImbalancedDatasetSampler 추가
    if hyp is None:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    else:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset)
            if rank != -1
            else ImbalancedDatasetSampler(dataset)
            if hyp["imbalanced"] and augment
            else None
        )
    #######################################

    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ", end=""
            )

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            print(f"image {self.count}/{self.nf} {path}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe="0", img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f"Camera Error {self.pipe}"
        img_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources="streams.txt", img_size=640, stride=32, auto=True):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            if "youtube.com/" in s or "youtu.be/" in s:  # if source is YouTube video
                check_requirements(("pafy", "youtube_dl"))
                import pafy

                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print("")  # newline

        # check for common shapes
        s = np.stack(
            [
                letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape
                for x in self.imgs
            ]
        )
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = (
            0,
            self.frames[i],
            1,
        )  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [
            letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0]
            for x in img0
        ]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep,
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        dist=-1,
        save_dir=None,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (
            self.augment and not self.rect
        )  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        self.save_dir = save_dir

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, "r") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x for x in t
                        ]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f"{prefix}{p} does not exist")
            self.img_files = sorted(
                [x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS]
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}")

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")

        #######################################
        # TODO: cache 삭제 - 데이터셋 충돌 방지
        if os.path.exists(cache_path):
            os.remove(cache_path)
        #######################################

        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == 0.4 and cache["hash"] == get_hash(
                self.label_files + self.img_files
            )
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache["msgs"]:
                logging.info("\n".join(cache["msgs"]))  # display warnings
        assert (
            nf > 0 or not augment
        ), f"{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
            )

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            # if cache_images == 'disk':
            #     self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
            #     self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
            #     self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(
                lambda x: load_image(*x), zip(repeat(self), range(n))
            )
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == "disk":
                    pass
                    # if not self.img_npy[i].exists():
                    #     np.save(self.img_npy[i].as_posix(), x[0])
                    # gb += self.img_npy[i].stat().st_size
                else:
                    (
                        self.imgs[i],
                        self.img_hw0[i],
                        self.img_hw[i],
                    ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})"
            pbar.close()

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))
                ),
                desc=desc,
                total=len(self.img_files),
            )
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info("\n".join(msgs))
        if nf == 0:
            logging.info(f"{prefix}WARNING: No labels found in {path}. See {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["msgs"] = msgs  # warnings
        x["version"] = 0.4  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            logging.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            logging.info(
                f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}"
            )  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            #######################################
            # TODO: bbox-cutmix augmentation
            labels = self.labels[index].copy()
            if hyp is not None:
                if hyp["bbox_cutmix"] and self.augment:
                    img, labels = bbox_cutmix(self, img, labels, h, w, index)
            #######################################

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
                )

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
            )

        if self.augment:
            # Albumentations
            if random.random() < hyp["more_aug"]:
                img, labels = self.albumentations(img, labels)
                nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(
                    img[i].unsqueeze(0).float(),
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                )[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat(
                    (torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2
                )
                l = (
                    torch.cat(
                        (label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0
                    )
                    * s
                )
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
#########################################################################
def save_image(obj, lb, name, p=None):
    h = obj.shape[0]
    w = obj.shape[1]

    img = obj.copy()
    if p is not None:
        for bbox in p:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if lb is not None:
            for bbox in lb:
                bbox = [
                    0,
                    (bbox[1] - bbox[3] / 2) * w,
                    (bbox[2] - bbox[4] / 2) * h,
                    bbox[3] * w,
                    bbox[4] * h,
                ]
                bbox = [int(x) for x in bbox]
                img = cv2.rectangle(
                    img, (bbox[1], bbox[2]), (bbox[1] + bbox[3], bbox[2] + bbox[4]), (0, 0, 255), 2
                )
    else:
        if lb is not None:
            lb = [int(x) for x in lb]
            img = cv2.rectangle(
                img, (lb[1], lb[2]), (lb[1] + lb[3], lb[2] + lb[4]), (0, 0, 255), 2
            )
    cv2.imwrite(f"{name}.jpg", img)


def bbox_cutmix(self, img, labels, h, w, index):
    # for logging
    os.makedirs(f"{self.save_dir}/img", exist_ok=True)

    # lables = [class, x,y,w,h]
    # h, w = height, width of img
    xmin = int(min(labels[:, 1] - labels[:, 3] / 2) * w)
    xmax = int(max(labels[:, 1] + labels[:, 3] / 2) * w)
    ymin = int(min(labels[:, 2] - labels[:, 4] / 2) * h)
    ymax = int(max(labels[:, 2] + labels[:, 4] / 2) * h)

    # crop-obj 넣을 빈 공간, 8곳
    # 1 2 3
    # 4 5 6
    # 7 8 9
    Position = [
        [0, 0, xmin, ymin],  # 1
        [xmin, 0, xmax, ymin],  # 2
        [xmax, 0, w, ymin],  # 3
        [0, ymin, xmin, ymax],  # 4
        # 5는 실제 obj
        [xmax, ymin, w, ymax],  # 6
        [0, ymax, xmin, h],  # 7
        [xmin, ymax, xmax, h],  # 8
        [xmax, ymax, w, h],  # 9
    ]

    # 구역마다 이미지 넣기
    for p in Position:
        if torch.rand(1) > 0.7:  # crop aug 확률 추가
            continue
        # 1. crop-obj 생성
        load_idx = torch.randint(len(self.indices), (1,)).item()  # crop할 이미지 선정
        load_img, a, (h_l, w_l) = load_image(self, load_idx)
        try:
            lb = self.labels[load_idx].copy()[
                torch.randint(len(self.labels[load_idx]), (1,)).item()
            ]
        except:
            print("-" * 30)
            print(">> bbox_cutmix - index error")
            print(f" - load index : {load_idx}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - file path : {self.img_files[load_idx]}")
            print("-" * 30)
            exit()

        xmin_load = int((lb[1] - lb[3] / 2) * w_l)
        xmax_load = int((lb[1] + lb[3] / 2) * w_l)
        ymin_load = int((lb[2] - lb[4] / 2) * h_l)
        ymax_load = int((lb[2] + lb[4] / 2) * h_l)

        # 랜덤 크기의 offset 생성(4방향 다 다름)
        offset = torch.randint(
            low=int(self.img_size / 40), high=int(self.img_size / 20), size=(4,)
        ).numpy()
        offset_xmin = np.clip(min(xmin_load, offset[0]), 0, offset[0])
        offset_xmax = np.clip(min(w_l - xmax_load, offset[1]), 0, offset[1])
        offset_ymin = np.clip(min(ymin_load, offset[2]), 0, offset[2])
        offset_ymax = np.clip(min(h_l - ymax_load, offset[3]), 0, offset[3])
        # normalize of load_img

        # 실제 crop된 이미지
        cropped_object = load_img[
            ymin_load - offset_ymin : ymax_load + offset_ymax,
            xmin_load - offset_xmin : xmax_load + offset_xmax,
        ].copy()

        # crop-object 크기
        height = cropped_object.shape[0]
        width = cropped_object.shape[1]

        # crop 되기 전,후 크기 비율 곱
        # class, x, y, w, h
        cropped_label = [
            lb[0],
            offset_xmin,
            offset_ymin,
            xmax_load - xmin_load,
            ymax_load - ymin_load,
        ]

        # bbox check
        # self.save_image(img.copy(), None, f'img/{index}_input')
        # self.save_image(cropped_object, cropped_label, f'img/{load_idx}_crop_1')

        # 2. crop-obj albumentations
        # FIXME:augment 추가 여부
        # crop-obj augmentation
        album_aug = A.Compose(
            [
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.ColorJitter(p=0.7),
                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=5, p=0.7),
                A.RandomRotate90(p=0.7),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["class_labels"], min_visibility=0.5
            ),
        )
        try:
            transformed = album_aug(
                image=cropped_object, bboxes=[cropped_label[1:]], class_labels=[cropped_label[0]]
            )
        except:
            print("-" * 30)
            print(">> bbox_cutmix - aug error")
            print(f" - bbox coord : {cropped_label[1:]}")
            print(f" - bbox size(HxW) : {lb[4]*h_l} x {lb[3]*w_l}")
            print(f" - image size(HxW) : {height} x {width}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - file path : {self.img_files[load_idx]}")
            print("-" * 30)
            save_image(cropped_object, cropped_label, f"{self.save_dir}/img/error_{load_idx}_aug")
            continue

        if len(transformed["class_labels"]) == 0:
            continue

        # augment 된 crop-obj
        cropped_object = transformed["image"]
        # bboxes = [(x,y,w,h), (x,y,w,h), ... ]
        # class_labels = [ c, c, ... ]
        cropped_label = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])
        # crop-object 크기
        height = cropped_object.shape[0]
        width = cropped_object.shape[1]

        # bbox check
        # self.save_image(cropped_object, cropped_label, f'img/{load_idx}_crop_2_album')

        # 3. img에 넣을 위치 정하기
        x_img = int(0.5 * torch.rand((1,)).item() * (p[2] - p[0])) + p[0]
        y_img = int(0.5 * torch.rand((1,)).item() * (p[3] - p[1])) + p[1]

        # 4. crop-aug 된 img 생성
        # p구역에 이미지 넣지 못 할 때
        if p[2] - x_img <= width or p[3] - y_img <= height:
            # FIXME:threshold 값 정하기
            # 최대 offset 평균 : 200(100x2), min object size : 20
            # -> 200 x 200 이상인 경우에만 resize 적용
            # -> self.img_size/10 x self.img_size/10 이상인 경우에만 resize 적용
            if p[2] - x_img > self.img_size / 10 and p[3] - y_img > self.img_size / 10:
                # 더 튀어나온 방향 기준으로 scale 정함
                scale_rate = min((p[2] - x_img) / width, (p[3] - y_img) / height)

                album_resize = A.Compose(
                    [A.Resize(int(height * scale_rate), int(width * scale_rate))],
                    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
                )
                try:
                    transformed = album_resize(
                        image=cropped_object,
                        bboxes=[cropped_label[1:]],
                        class_labels=[cropped_label[0]],
                    )
                except:
                    print("-" * 30)
                    print(">> bbox_cutmix - resize error")
                    print(f" - crop coord : {cropped_label[1:]}")
                    print(f" - crop size(HxW) : {cropped_label[4]} x {cropped_label[3]}")
                    print(f" - image size(HxW) : {height} x {width}")
                    print(f" - label : {self.labels[load_idx]}")
                    print(f" - file path : {self.img_files[load_idx]}")
                    print("-" * 30)
                    save_image(
                        cropped_object,
                        cropped_label,
                        f"{self.save_dir}/img/error_{load_idx}_resize",
                    )
                    continue

                # resize 된 crop-obj
                cropped_object = transformed["image"]
                cropped_label = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])

                # resize 된 crop-object 크기
                height = cropped_object.shape[0]
                width = cropped_object.shape[1]

                # bbox check
                # self.save_image(cropped_object, cropped_label, f'img/{load_idx}_crop_3_resize')
            # 220 x 220 이하인 경우 다음 구역으로
            else:
                continue

        # 좌표 수정
        # bbox_coords = [0, x_img*w+cropped_label[1], y_img*h+cropped_label[2], cropped_label[3], cropped_label[4]]
        cropped_label[1] = (
            x_img + cropped_label[1] + cropped_label[3] / 2
        ) / w  # (crop 이미지 넣을 좌표 + offset 크기 + 넓이/2)
        cropped_label[2] = (y_img + cropped_label[2] + cropped_label[4] / 2) / h
        cropped_label[3] = cropped_label[3] / w
        cropped_label[4] = cropped_label[4] / h

        # label 추가
        labels = np.vstack((labels, np.array([cropped_label])))

        # img에 crop-obj 삽입
        try:
            img[y_img : y_img + height, x_img : x_img + width] = cropped_object
        except:
            print("-" * 30)
            print(">> bbox_cutmix - paste error")
            print(f" - crop coord : {cropped_label[1:]}")
            print(f" - crop size(HxW) : {height} x {width}")
            print(f" - min point : ({y_img}, {x_img})")
            print(f" - max point : ({y_img+height}, {x_img+width})")
            print(f" - image size(HxW) : {img.shape[1]} x {img.shape[0]}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - crop image path : {self.img_files[load_idx]}")
            print(f" - file path : {self.img_files[index]}")
            print("-" * 30)
            save_image(
                img.copy(),
                labels,
                f"{self.save_dir}/img/error_{load_idx}_{index}_paste",
                p=Position,
            )
            continue

        # label 추가
        # labels = np.vstack((labels, np.array([cropped_label])))

        # bbox check
        if index in range(10, 101, 10):
            # print("bbox check", img.shape, index)
            save_image(img.copy(), labels, f"{self.save_dir}/img/monitor_{index}", p=Position)

    return img, labels


#########################################################################


def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        # npy = self.img_npy[i]
        # if npy and npy.exists():  # load npy
        #     im = np.load(npy)
        # else:  # read image
        #     path = self.img_files[i]
        #     im = cv2.imread(path)  # BGR
        #     assert im is not None, 'Image Not Found ' + path
        path = self.img_files[i]
        im = cv2.imread(path)  # BGR
        assert im is not None, "Image Not Found " + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i].copy(), self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    hyp = self.hyp

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        #######################################
        # TODO: bbox-cutmix augmentation
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if hyp is not None:
            if hyp["bbox_cutmix"] and self.augment:
                img, labels = bbox_cutmix(self, img, labels, h, w, index)
        #######################################

        # place img in img4
        if i == 0:  # top left
            img4 = np.full(
                (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                max(yc - h, 0),
                xc,
                yc,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        # labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], w, h, padw, padh
            )  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
    img4, labels4 = random_perspective(
        img4,
        labels4,
        segments4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    hyp = self.hyp

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        #######################################
        # TODO: bbox-cutmix augmentation
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if hyp is not None:
            if hyp["bbox_cutmix"] and self.augment:
                img, labels = bbox_cutmix(self, img, labels, h, w, index)
        #######################################

        # place img in img9
        if i == 0:  # center
            img9 = np.full(
                (s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        # labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], w, h, padx, pady
            )  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(
        img9,
        labels9,
        segments9,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )  # border to remove

    return img9, labels9


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path="../datasets/coco128"):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + "_flat")
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + "/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path="../datasets/coco128"):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / "classifier") if (
        path / "classifier"
    ).is_dir() else None  # remove existing
    files = list(path.rglob("*.*"))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, "r") as f:
                    lb = np.array(
                        [x.split() for x in f.read().strip().splitlines()], dtype=np.float32
                    )  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (
                        (path / "classifier") / f"{c}" / f"{path.stem}_{im_file.stem}_{j}.jpg"
                    )  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1] : b[3], b[0] : b[2]]), f"box failure in {f}"


def autosplit(path="../datasets/coco128/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum(
        [list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], []
    )  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(
        f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only
    )
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(
                    "./" + img.relative_to(path.parent).as_posix() + "\n"
                )  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = (
        0,
        0,
        0,
        0,
        "",
        [],
    )  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    pass
                    # Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)  # re-save image
                    # msg = f'{prefix}WARNING: corrupt JPEG restored and saved {im_file}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, "r") as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l
                    ]  # (cls, xy1...)
                    l = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, "labels require 5 columns each"
                assert (l >= 0).all(), "negative labels"
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                assert np.unique(l, axis=0).shape[0] == l.shape[0], "duplicate labels"
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(
    path="coco128.yaml", autodownload=False, verbose=False, profile=False, hub=False
):
    """Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith(".zip"):  # path is data.zip
            assert Path(path).is_file(), f"Error unzipping {path}, file not found"
            assert os.system(f"unzip -q {path} -d {path.parent}") == 0, f"Error unzipping {path}"
            dir = path.with_suffix("")  # dataset directory
            return True, str(dir), next(dir.rglob("*.yaml"))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f'
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(im_dir / Path(f).name, quality=75)  # save

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors="ignore") as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data["path"] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data["path"] + ("-hub" if hub else ""))
    stats = {"nc": data["nc"], "names": data["names"]}  # statistics dictionary
    for split in "train", "val", "test":
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc="Statistics"):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data["nc"]))
        x = np.array(x)  # shape(128x80)
        stats[split] = {
            "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
            "image_stats": {
                "total": dataset.n,
                "unlabelled": int(np.all(x == 0, 1).sum()),
                "per_class": (x > 0).sum(0).tolist(),
            },
            "labels": [
                {str(Path(k).name): round_labels(v.tolist())}
                for k, v in zip(dataset.img_files, dataset.labels)
            ],
        }

        if hub:
            im_dir = hub_dir / "images"
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(
                ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files),
                total=dataset.n,
                desc="HUB Ops",
            ):
                pass

    # Profile
    stats_path = hub_dir / "stats.json"
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix(".npy")
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f"stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write")

            file = stats_path.with_suffix(".json")
            t1 = time.time()
            with open(file, "w") as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, "r") as f:
                x = json.load(f)  # load hyps dict
            print(f"stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write")

    # Save, print and return
    if hub:
        print(f"Saving {stats_path.resolve()}...")
        with open(stats_path, "w") as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
