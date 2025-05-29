# Transfer Learning Practice: ResNet & R-CNN

This repository contains code and configuration files to reproduce fine-tuning experiments on Caltech-101 classification (ResNet-18) and Pascal VOC object detection/segmentation (Mask R-CNN & Sparse R-CNN) using MMDetection.

------

## Repository Structure

```
|   README.md
|
+---mmdetection
|   |   Mask-RCNN.ipynb
|   |   Sparse-RCNN.ipynb
|   |
|   +---configs
|   |       coco_instance.py
|   |       schedule_1x.py
|   |       voc0712.py
|   |       voc_instance.py
|   |       voc_mask_rcnn_r50_fpn.py
|   |       voc_sparse_rcnn_r50_fpn.py
|   |
|   +---Mask_RCNN
|   |
|   +---Sparse_RCNN
|   |
|   \---work_dirs
|       +---voc_mask_rcnn_r50_fpn
|       |
|       \---voc_sparse_rcnn_r50_fpn
|
\---Resnet
    |   config.py
    |   dataset.py
    |   data_split.py
    |   model.py
    |   Resnet Finetune.md
    |   test.py
    |   train.py
    |   train_resnet_compare.py
    |
    +---checkpoints
    +---data
    |   \---caltech-101
    |       +---101_ObjectCategories
    |       \---Annotations
    +---runs
    |   \---caltech101
    |           
    |
    \---splits
            splits.json
    
```

------

## Introduction

## ResNet-18 Fine-Tuning (Caltech-101)

1. **Prepare data**
   Download Caltech-101 and place contents under `Resnet/data/caltech-101/101_ObjectCategories`.

2. **Generate splits**
   Use or adapt `Resnet/splits/train.txt`, `val.txt`, `test.txt` for 30/10/remaining splits per class.

3. **Train**

   ```
   python train_resnet.py \
     --data-dir Resnet/data/caltech-101 \
     --splits-dir Resnet/splits \
     --output-dir Resnet/runs/caltech101 \
     --backbone-lr 1e-4 \
     --head-lr 1e-2 \
     --batch-size 32 \
     --epochs 80
   ```

4. **Evaluate**

   ```
   python evaluate_resnet.py \
     --checkpoint Resnet/runs/caltech101/best.pth \
     --data-dir Resnet/data/caltech-101 \
     --splits-dir Resnet/splits
   ```

------

## Mask R-CNN & Sparse R-CNN (Pascal VOC)

All configurations reside under `mmdetection/configs/Mask_RCNN` and `.../Sparse_RCNN`.

### 1. Prepare VOC Dataset

Download and extract `VOC2007` & `VOC2012` into `mmdetection/data/VOCdevkit/`.

### 2. Training

Mask R-CNN:

```
cd mmdetection
python tools/train.py \
  configs/Mask_RCNN/voc_mask_rcnn_r50_fpn.py \
  --work-dir work_dirs/voc_mask_rcnn_r50_fpn
```

Sparse R-CNN:

```
python tools/train.py \
  configs/Sparse_RCNN/voc_sparse_rcnn_r50_fpn.py \
  --work-dir work_dirs/voc_sparse_rcnn_r50_fpn
```

### 3. Evaluation

Mask R-CNN:

```
python tools/test.py \
  configs/Mask_RCNN/voc_mask_rcnn_r50_fpn.py \
  work_dirs/voc_mask_rcnn_r50_fpn/latest.pth \
  --eval mAP
```

Sparse R-CNN:

```
python tools/test.py \
  configs/Sparse_RCNN/voc_sparse_rcnn_r50_fpn.py \
  work_dirs/voc_sparse_rcnn_r50_fpn/latest.pth \
  --eval mAP
```

------

## Results & Logs

- **ResNet-18**: best test accuracy ~87.3 % .
- **Mask R-CNN**: bbox mAP ≈ 45 %, segm mAP ≈ 45 % .
- **Sparse R-CNN**: bbox mAP ≈ 70 % .

## Checkpoints

Model Checkpoints can be downloaded at https://drive.google.com/drive/folders/1xzAAHd5RznnsV8IO3EO2dn91BzQ40yA3?usp=drive_link
