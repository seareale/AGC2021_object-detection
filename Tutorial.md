# Recyclable Objects Detection
This document is for the project: Development of Recyclable Objects Detection Model. For more information or any questions, please contact to seareale@gmail.com

### Table of contents
- [How to start](#how-to-start)
	* [1. Environments](#1-environments)
	* [2. Run](#2-run)
- [History](#history)
- [Experiments](#experiments)
	* [1. Bounding box CutMix](#1-bounding-box-cutmix)
	* [2. Standardized Distance-based IoU](#2-standardized-distance-based-iou)
- [Checkpoints](#checkpoints)
- [Reference](#reference)

## <div align="center">How to start</div>
### 1. Environments
1) create a conda environment and install YOLOv5[[1]](https://github.com/ultralytics/yolov5).
```bash
$ conda create -n recycle python=3.8
$ conda activate recycle
$ git clone https://github.com/ultralytics/yolov5.git # commit hash : 47233e1698b89fc437a4fb9463c815e9171be955
$ cd yolov5
$ pip install -r requirements.txt
```
2) prepare BboxCutMix[[2]](https://github.com/seareale/BboxCutMix) and SDIoU[[3]](https://github.com/seareale/SDIoU).
```bash
$ mkdir custom && cd custom
$ git clone https://github.com/seareale/BboxCutMix.git
$ cd BboxCutMix && pip install -r requirements.txt && cd ..
$ git clone https://github.com/seareale/SDIoU.git
$ cd SDIoU && pip install -r requirements.txt && cd ..
```
3) create `datasets.yaml` & `hyp.yaml` in the path `./custom`. 
```yaml
# datasets.yaml
path : '/ssd_2/RecycleTrash/images/' # 60 server
train: [
        # 2020AGC
        '201028_kvl',
        # '201031_data', 
        # '201103_data', '201107_data', '201110_data', '201114_data', 
        # '201117_data', '201118_data', '201127_data',
        # '201209_last',

        # follow-up study
        # '210727_easy', '210727_hard',
        # '210809_easy', '210809_hard',
        # '210901_easy', '210901_hard',
        # '201208_img', '210906_easy', '210906_hard',

        # 2021AGC
        # '211014_data_re', '211021_data', '211022_data', '211025_data',
        # '211026_data', '211027_data', '211028_data', '211029_data', 
        # '211030_data', '211031_data', 
        # '211101_data', '211102_data', '211103_data', '211104_data',

        # test data
        # 'test_2021AGC_30', # 2021AGC sample data
        # 'test_211026_405', 'test_211103_400' # test data from seoreu
        ]
val: [
        'test' # 2020AGC sample data(labeled partly)
        # 'test_2020AGC_452', # 2020AGC sample data(labeled as a whole)
        ]
nc: 7
names: ['paper','paperpack','can','glass','pet','plastic','vinyl']
```

```yaml
# hyp.yaml

...  # after any hyp file
# recommand hyp.scratch-med.yaml in 'data/hyps'

bbox_cutmix: True
iou_method: 'ciou' # 'ciou', 'diou', 'sdiou'
std_type: None # None, 'mean', 'var' (use this when iou_method is 'sdiou')
```
4) modify `utils/dataloaders.py` and `loss.py`.
```python
# utils/dataloaders.py
...

from custom.BboxCutMix.bbox_cutmix import bbox_cutmix

...

class LoadImagesAndLabels(Dataset):
	...
	def __getitem__(self, index):
    	...
		else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            
            labels = self.labels[index].copy() # moved
            ################################################
            # BboxCutMix
            if hyp is not None:
                if hyp["bbox_cutmix"] and self.augment:
                    img, labels = bbox_cutmix(self, img, labels, h, w)
            ################################################
            ...
            
	def def load_mosaic(self, index):
 		...
        # Load image
        img, _, (h, w) = self.load_image(index)
        
        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy() # moved
        ################################################
        # BboxCutMix
        hyp = self.hyp
        if hyp is not None:
          if hyp["bbox_cutmix"] and self.augment:
            img, labels = bbox_cutmix(self, img, labels, h, w)
        ################################################
```

```python
# utils/loss.py
...

# from utils.metrics import bbox_iou
from custom.SDIoU.sdiou import bbox_iou

...

class ComputeLoss:
	...
	def __call__(self, p, targets):
    ...
    # iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze() # iou(prediction, target)
    #########################################################
    # SDIoU
    if self.hyp["iou_method"] == "diou":
    	iou = bbox_iou(pbox, tbox[i], DIoU=True).squeeze()
    elif self.hyp["iou_method"] == "ciou":
    	iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
    elif self.hyp["iou_method"] == "sdiou":
    	iou = bbox_iou(pbox, tbox[i], SDIoU=True, std_type=self.hyp["std_type"]).squeeze()
	#########################################################

```

### 2. Run
1) DP(DataParallel Mode) - 8 GPU
```bash
$ python3 train.py --epochs 30 --batch 32 --img-size 1280 \
--device 0,1,2,3,4,5,6,7 \
--hyp custom/hyp.yaml --data custom/datasets.yaml \
--weights yolov5x6.pt
```

2) DDP(DistributedDataParallel Mode) - 8 GPU
```bash
$ python3 -m torch.distributed.launch --nproc_per_node 8 \
--master_port {PORT_NUMBER} \
train.py --epochs 30 --batch 32 --img-size 1280 \
--device 0,1,2,3,4,5,6,7 --worker 32 \
--hyp custom/hyp.yaml --data custom/datasets.yaml \
--weights yolov5x6.pt
```

## <div align="center">History</div>
> **[2020AGC(AI Grand Challenge)](http://www.ai-challenge.kr) (Nov 16, 2020  ~ Nov 20, 2020)**  
**🥈 2nd Place Winner** of Object Classification Track  
Based on [mmdetection](https://github.com/open-mmlab/mmdetection), Cascade R-CNN acheived macro F1 score 0.8524.
>- source_1 : [https://github.com/jaebbb/Recycle-Trash-Detection](https://github.com/jaebbb/Recycle-Trash-Detection)  
>- source_2 : [https://github.com/seareale/recycle-detection](https://github.com/seareale/recycle-detection)

> **Follow-up study (Nov 20, 2020  ~ Dec 31, 2021)**   
Used one-stage detector YOLOv5 for embeded and improved performance in various methods.
>- A total of 130,000 training image datasets
>- **TensorRT**(Quantization)[[4]](https://github.com/heechul-knu/yolov5-recycle-tensorrt)
>- **Pruning** for embedded systems(use Global pruning)[[5]](https://github.com/heechul-knu/yolov5-slimming)[[6]](https://github.com/heechul-knu/yolov5-recycle-pruning)
>- **Multi transfer learning**(COCO -> Object365 -> easy background -> hard background)
>- **BboxCutMix**(Bounding box CutMix)[[2]](https://github.com/seareale/BboxCutMix) for solving problems of Mosaic
>- Final macro F1 score is 0.91 (mAP@0.5 : 0.94)

> **[2021AGC(AI Grand Challenge)](http://www.ai-challenge.kr) (Nov 2021)**  
**4th Place** of Object Classification Track
>- 1st : (주)딩브로
>- 2nd : 튜닙
>- 3rd : (주)이스트소프트

> **Study for thesis (Apr, 2022  ~ May, 2022)**   
Found out the problems of DIoU and developed SDIoU(Standardized Distance-based IoU).
>- source : [https://github.com/seareale/SDIoU](https://github.com/seareale/SDIoU)

## <div align="center">Experiments</div>
The following experiments were conducted with 70,000 images, not the total dataset(130,000 images) and used the COCO pretrained weight(yolov5x6.pt).
### 1. Bounding box CutMix
<div align='center' style='column-count: 1;'>

<p>Results of Recyclable Objects dataset</p>

Method                 | mAP@0.5 | mAP@0.5:0.95
|:-|:-:|:-:|
No Mosaic                 | 0.89508 | 0.77857 
Base(Mosaic)             | 0.91090 | 0.79401
\+ Mixup                   | 0.91700 | 0.80490 
\+ copy-paste              | 0.91668 | 0.79978
\+ Mixup + copy-paste      | 0.91291 | 0.79952 
\+ BboxCutMix              | 0.91967 | 0.80758
\+ BboxCutMix + Mixup      | 0.91890 | **0.80798**
\+ BboxCutMix + copy-paste | **0.91978** | 0.80622
\+ All                     | 0.90412 | 0.79141
  
</div>


### 2. Standardized Distance-based IoU
<div align='center' style='column-count: 2;'>

<p>Results of Recyclable Objects dataset</p>
  
Method          | mAP@0.5 | mAP@0.5:0.95
|:-|:-:|:-:|
Base(CIoU)      | 0.91090 | 0.79401
DIoU            | 0.90234 | 0.78660
SDIoU(mean)     | **0.91149** | **0.79759**
SDIoU(variance) | 0.90825 | 0.79313

<p>Results of PASCAL VOC</p>
  
Method | mAP@0.5 | mAP@0.5:0.95
|:-|:-:|:-:|
DIoU            | 0.88758 | 0.68262
SDIoU(mean)     | 0.88900 | **0.68473**
SDIoU(variance) | **0.89029** | 0.68335
  
</div>

## <div align="center">Checkpoints</div>
```bash
# In 60 server, '/ssd_2/RecycleTrash/yolov5-weights/'
├── agc2020 # trained with agc2020 dataset
│   ├── for_paper # deprecated
│   ├── recycle_l6_last
│   ├── recycle_m6_last
│   ├── recycle_s6_last
│   └── recycle_x6_last
├── agc2021 
	# init neck or detector with COCO weights (backbone - neck - detector)
	# trained with all Recyclable Objects dataset, used agc2020 weights
│   ├── agc2021_l6_last_all 
│   ├── agc2021_l6_last_neck-det 
│   ├── agc2021_m6_last_coco-neck-det
│   ├── agc2021_s6_last_coco-neck-det
	# trained with agc2021 dataset, used agc2020 weights
│   ├── agc2021_x6_last_all_last
│   ├── agc2021_x6_last_coco-neck-det
│   └── agc2021_x6_last_coco-neck-det_last
└── pretrained
│   ├── object365_yolov5s6_epoch01.pt # trained with Object365 
│   ├── object365_yolov5s6_epoch25.pt
│   ├── object365_yolov5s6_epoch30.pt
│   ├── object365_yolov5s6_epoch48.pt
│   ├── object365_yolov5x6_epoch14_346.pt
    # trained with all Recyclable Objects dataset, used agc2020 weights
│   ├── recycle_m6_last_all.pt
│   ├── recycle_m6_last_coco_det.pt # trained with COCO
│   ├── recycle_m6_last_coco_neck-det.pt
│   ├── recycle_s6_last_coco_det.pt
│   ├── recycle_s6_last_coco_neck-det.pt
│   ├── recycle_s6_last_rand_det.pt # init random weights
│   ├── recycle_s6_last_rand_neck-det.pt
    # trained with agc2021 dataset, used agc2020 weights
│   ├── recycle_x6_bboxcutmix_all.pt
│   ├── recycle_x6_bboxcutmix_coco_det.pt
│   ├── recycle_x6_bboxcutmix_coco_neck-det.pt
│   ├── recycle_x6_bboxcutmix_rand_det.pt
│   ├── recycle_x6_bboxcutmix_rand_neck-det.pt
│   ├── yolov5l6.pt # from YOLOv5
│   ├── yolov5m6.pt
│   ├── yolov5s6.pt
│   └── yolov5x6.pt
└── thesis
    ├── exp1
    │   ├── bboxcutmix
    │   ├── exp1_base
    │   ├── exp1.yaml
    │   └── sdiou
    │       ├── recycle
    │       └── VOC_pretrained_m6
    └── exp2
        ├── bboxcutmix # No experiments : All, bboxcutmix + copy-paste
        ├── exp2_base
        ├── exp2.yaml
        └── sdiou # Empty
```

## Reference
1. YOLOv5, https://github.com/ultralytics/yolov5
2. BboxCutMix, https://github.com/seareale/BboxCutMix
3. SDIoU, https://github.com/seareale/SDIoU
4. TensorRT experiments, https://github.com/heechul-knu/yolov5-recycle-tensorrt
5. Pruning experiments 1, https://github.com/heechul-knu/yolov5-slimming
6. Pruning experiments 2, https://github.com/heechul-knu/yolov5-recycle-pruning
