# Written by @kentaroy47

import albumentations.augmentations.transforms as A
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import asnumpy, rearrange
from .native import MedianPool2d, SameAvg2D

from .native import MedianPool2d, SameAvg2D


def albumentations(func):
    """
    For albumentations, conver torch tensor (c h w) to numpy (h w c)
    """

    def new_func(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = torch.device(arg.device)
                new_arg = rearrange(asnumpy(arg), "... c h w -> ... h w c")
                new_args.append(new_arg)
            else:
                new_args.append(arg)
        output = func(*new_args, **kwargs)
        output = rearrange(output, "... h w c -> ... c h w")
        return torch.tensor(output).to(device)

    return new_func


class Base:
    def augment(self, image):
        # pass torch tensors
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        """
        boxes format [x1, y1, x2, y2]
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class TorchMedianBlur(Base):
    def __init__(self, kernel_size=7):
        self.median = MedianPool2d(kernel_size, same=True)

    def augment(self, image):
        image = image.unsqueeze(0)
        return self.median(image).squeeze(0)

    def batch_augment(self, images):
        return self.median(images)

    def deaugment_boxes(self, boxes):
        return boxes


class TorchBlur(Base):
    def __init__(self, kernel_size=7):
        self.blur = SameAvg2D(kernel_size, same=True)

    def augment(self, image):
        image = image.unsqueeze(0)
        return self.blur(image).squeeze(0)

    def batch_augment(self, images):
        return self.blur(images)

    def deaugment_boxes(self, boxes):
        return boxes


class RandColorJitter(Base):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def augment(self, image):
        return self.jitter(image)

    def batch_augment(self, images):
        return self.jitter(images)

    def deaugment_boxes(self, boxes):
        return boxes


class Blur(Base):
    def __init__(self, blur_limit=7) -> None:
        self.blur = A.Blur(blur_limit=blur_limit, p=1)

    @albumentations
    def augment(self, image):
        return self.blur(image=image)["image"]

    def batch_augment(self, images):
        images = [self.augment(image) for image in images]
        images = torch.stack(images, axis=0)
        return images

    def deaugment_boxes(self, boxes):
        return boxes


class MotionBlur(Base):
    def __init__(self, blur_limit=7) -> None:
        self.motionblur = A.MotionBlur(blur_limit=blur_limit, p=1)

    @albumentations
    def augment(self, image):
        return self.motionblur(image=image)["image"]

    def batch_augment(self, images):
        images = [self.augment(image) for image in images]
        images = torch.stack(images, axis=0)
        return images

    def deaugment_boxes(self, boxes):
        return boxes


class MedianBlur(Base):
    def __init__(self, blur_limit=5) -> None:
        self.medianblur = A.MedianBlur(blur_limit=blur_limit, p=1)

    @albumentations
    def augment(self, image):
        return self.medianblur(image=image)["image"]

    def batch_augment(self, images):
        images = [self.augment(image) for image in images]
        images = torch.stack(images, axis=0)
        return images

    def deaugment_boxes(self, boxes):
        return boxes


class HorizontalFlip(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return image.flip(1)

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.imsize - boxes[:, [3, 1]]
        return boxes


class VerticalFlip(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return image.flip(2)

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.imsize - boxes[:, [2, 0]]
        return boxes


class Rotate90Left(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.imsize - boxes[:, [3, 1]]
        res_boxes[:, [1, 3]] = boxes[:, [0, 2]]
        return res_boxes


class Rotate90Right(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return torch.rot90(image, 1, (2, 1))

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return torch.rot90(images, 1, (3, 2))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [1, 3]] = self.imsize - boxes[:, [2, 0]]
        res_boxes[:, [0, 2]] = boxes[:, [3, 1]]
        return res_boxes


class Multiply(Base):
    # change brightness of image
    def __init__(self, scale):
        # scale is a float value 0.5~1.5
        self.scale = scale

    def augment(self, image):
        return image * self.scale

    def batch_augment(self, images):
        return images * self.scale

    def deaugment_boxes(self, boxes):
        return boxes


class MultiScale(Base):
    # change scale of the image for TTA.
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale

    def augment(self, image):
        return F.interpolate(image, scale_factor=self.imscale, recompute_scale_factor=True)

    def batch_augment(self, images):
        return F.interpolate(images, scale_factor=self.imscale, recompute_scale_factor=True)

    def deaugment_boxes(self, boxes):
        return boxes / self.imscale

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.imscale})"


class MultiScaleFlip(Base):
    # change scale of the image and hflip.
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale

    def augment(self, image):
        self.imsize = image.shape[1]
        return F.interpolate(image, scale_factor=self.imscale, recompute_scale_factor=True).flip(2)

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return F.interpolate(images, scale_factor=self.imscale, recompute_scale_factor=True).flip(
            3
        )

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.imsize * self.imscale - boxes[:, [2, 0]]
        boxes = boxes / self.imscale
        return boxes


class MultiScaleHFlip(Base):
    # change scale of the image and vflip.
    # not useful for 2d detectors..
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale

    def augment(self, image):
        self.imsize = image.shape[1]
        return F.interpolate(image, scale_factor=self.imscale, recompute_scale_factor=True).flip(1)

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return F.interpolate(images, scale_factor=self.imscale, recompute_scale_factor=True).flip(
            2
        )

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.imsize * self.imscale - boxes[:, [2, 0]]
        boxes = boxes / self.imscale
        return boxes


class TTACompose(Base):
    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

    def __repr__(self):
        return str(self.transforms)


from .nms import nms, soft_nms
from .wbf import weighted_boxes_fusion


class nms_func:
    """
    class to call nms during inference.
    """

    def __init__(self, nmsname="wbf", weights=None, iou_thr=0.5, skip_box_thr=0.1):
        self.weights = weights
        self.iou = iou_thr
        self.skip = skip_box_thr
        self.nms = nmsname

    def __call__(self, boxes_list, scores_list, labels_list):
        if self.nms == "wbf":
            return weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, self.weights, self.iou, self.skip
            )
        elif self.nms == "nms":
            return nms(
                boxes_list, scores_list, labels_list, iou_thr=self.iou, weights=self.weights
            )
        # TODO: add soft-nms
        else:
            raise NotImplementedError()


# Model wrapper
class TTAWrapper:
    """
    wrapper for tta and inference.
    model: your detector. Right now, must output similar to the torchvision frcnn model.
    mono: tta which do not configure the image size.
    multi: tta which configures the image size.
    These two must be declared separetly.
    nms: choose what nms algorithm to run. right now, wbf or nms.
    iou_thr: iou threshold for nms
    skip_box_thr: score threshold for nms
    weights: for weighted box fusion, but None is fine.
    """

    def __init__(
        self,
        model,
        tta,
        scale=[1],
        nms="wbf",
        iou_thr=0.5,
        skip_box_thr=0.5,
        weights=None,
        device=torch.device("cpu"),
        half_flag=False,
    ):
        self.ttas = self.generate_TTA(tta, scale)
        self.model = model  # .eval()
        # set nms function
        # default is weighted box fusion.
        self.nms = nms_func(nms, weights, iou_thr, skip_box_thr)
        self.device = device
        self.half_flag = half_flag

    def generate_TTA(self, tta, scale):
        from itertools import product

        tta_transforms = []

        # Generate ttas for monoscale TTAs
        if len(scale) == 1 and scale[0] == 1:
            print("preparing tta for monoscale..")
            for tta_combination in product(*list([i, None] for i in tta)):
                tta_transforms.append(
                    TTACompose(
                        [tta_transform for tta_transform in tta_combination if tta_transform]
                    )
                )
        # Multiscale TTAs
        else:
            print("preparing tta for multiscale..")
            for s in scale:
                for tta_combination in product(*list([i, None] for i in tta)):
                    tta_transforms.append(
                        TTACompose(
                            [MultiScale(s)]
                            + [tta_transform for tta_transform in tta_combination if tta_transform]
                        )
                    )
        return tta_transforms

    def model_inference(self, img):
        with torch.no_grad():
            results = self.model(img)
        return results

    def tta_num(self):
        return len(self.ttas)

    def tta_inference(self, img):
        b_boxes = [[] for _ in range(img.shape[0])]
        b_scores = [[] for _ in range(img.shape[0])]
        b_labels = [[] for _ in range(img.shape[0])]
        # TTA loop
        for tta in self.ttas:
            # gen img
            inf_img = tta.batch_augment(img.clone())
            if self.half_flag:
                inf_img = inf_img.half()
            results = self.model_inference(inf_img.to(self.device))
            # iter for batch

            for idx, result in enumerate(results):
                boxes = result["boxes"].cpu().numpy()
                boxes = tta.deaugment_boxes(boxes)

                thresh = 0.01
                ind = result["scores"].cpu().numpy() > thresh

                b_boxes[idx].append(boxes[ind])
                b_scores[idx].append(result["scores"].cpu().numpy()[ind])
                b_labels[idx].append(result["labels"].cpu().numpy()[ind])
        return b_boxes, b_scores, b_labels

    def tta_nms(self, b_boxes, b_scores, b_labels):
        """
        NMS to different augmented images for tta
        """
        for i in range(len(b_boxes)):
            b_boxes[i], b_scores[i], b_labels[i] = self.nms(b_boxes[i], b_scores[i], b_labels[i])
        return b_boxes, b_scores, b_labels

    # TODO: change to call
    def __call__(self, img):
        b_boxes, b_scores, b_labels = self.tta_inference(img)
        b_boxes, b_scores, b_labels = self.tta_nms(b_boxes, b_scores, b_labels)
        return b_boxes, b_scores, b_labels


# for use in EfficientDets
class wrap_effdet:
    def __init__(self, model, imsize=512):
        # imsize.. input size of the model
        self.model = model
        self.imsize = imsize

    def __call__(self, img, score_threshold=0.22):
        # inference
        # TODO: Fix code
        det = self.model(img, torch.tensor([1] * images.shape[0]).float().cuda())

        predictions = []
        for i in range(img.shape[0]):
            # unwrap output
            boxes = det[i][:, :4]
            scores = det[i][:, 4]
            # filter output
            npscore = scores.detach().cpu().numpy()
            indexes = np.where(npscore > score_threshold)[0]
            boxes = boxes[indexes]
            # coco2pascal
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            # clamp boxes
            boxes = boxes.clamp(0, self.imsize - 1)
            # wrap outputs
            predictions.append(
                {
                    "boxes": boxes[indexes],
                    "scores": scores[indexes],
                    # TODO: update for multi-label tasks
                    "labels": torch.from_numpy(np.ones_like(npscore[indexes])).cuda(),
                }
            )

        return predictions


# for use in EfficientDets
class wrap_yolov5:
    def __init__(self, model_list, nms):
        # imsize.. input size of the model
        self.model_list = model_list
        self.nms = nms

    def __call__(
        self, img, conf_thres=0.2, iou_thres=0.6, agnostic_nms=False, multi_label=True, max_det=140
    ):
        # inference
        batch = None
        for model in self.model_list:
            if batch is None:
                batch = model(img)[0]
            else:
                batch += model(img)[0]
        batch /= len(self.model_list)

        batch = self.nms(
            batch,
            conf_thres,
            iou_thres,
            None,
            agnostic_nms,
            multi_label=multi_label,
            max_det=max_det,
        )

        predictions = []

        for pred in batch:
            predictions.append({"boxes": pred[:, :4], "scores": pred[:, 4], "labels": pred[:, 5]})

        return predictions