# AGC2021_object-detection
![](asset/agc2021_1.png)  
**[2021 AI Grand Challenge](https://www.ai-challenge.or.kr/) (Nov 10, 2021 ~ Nov 14, 2021)**  
- Object Detection Track  
<img src="asset/agc2021_2.png" width="50%"/>


## <div align="center">History</div>
> **[AI Grand Challenge](http://www.ai-challenge.kr) (Nov 16, 2020  ~ Nov 20,2020)**  
**🥈 2nd Place Winner** of Object Detection Track  
>- Based on [mmdetection](https://github.com/open-mmlab/mmdetection), Cascade R-CNN
>- Github : [https://github.com/jaebbb/Recycle-Trash-Detection](https://github.com/jaebbb/Recycle-Trash-Detection)

> **Follow-up study (Jan, 2021 ~ Dec, 2021)**  
>- Implementation of recyclable trash object detection model based on **YOLOv5**
>- Quantization, Pruning for embedded system
>- A total of **130,000** training image datasets
>- Final macro F1 score is **0.9** (mAP@.5 : 0.94)


## <div align="center">Environments</div>
```
Ubuntu 18.04   
CUDA 11.1.1
Python 3.8.3
```
Our object detection model is implemented in [YOLOv5](https://github.com/ultralytics/yolov5).


## <div align="center">Key points</div>
### Bbox Cutmix 
There were No-object images in training when using ***mosaic*** because of image center position of objects. So we implemented a augmentation pasting bboxes in the empty space of a training image(***Bbox Cumtix***).
<div align="center">
<img src="asset/image01.png" hspace=20/><img src="asset/image02.png" hspace=20/>
<p>An example of bbox cutmix</p>
</div>

```python
if self.crop_aug :
    img , labels = self.selfmix(img, labels, h, w)

...

def selfmix(self, img, labels, h, w):
    # get a bbox from a rand-selected img
    # augmentation for bbox
    # get coordinates from a training img to paste a bbox
        # resize a bbox(optional)
    # get a training img with a new bbox
    return img, labels
```

---
***continue***


### Transfer learning

### TTA

### Output post-processing



