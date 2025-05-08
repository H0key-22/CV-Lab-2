# Transfer Learning Benchmarks: ResNet & R-CNN

This repository contains code and configuration files to reproduce fine-tuning experiments on Caltech-101 classification (ResNet-18) and Pascal VOC object detection/segmentation (Mask R-CNN & Sparse R-CNN) using MMDetection.

------

## Repository Structure

```
├─ mmdetection
│  ├─ configs
│  │   ├─ Mask_RCNN
│  │   │   └─ voc_mask_rcnn_r50_fpn.py
│  │   ├─ Sparse_RCNN
│  │   │   └─ voc_sparse_rcnn_r50_fpn.py
│  │   └─ _base_ (common configs)
│  └─ work_dirs
│      ├─ voc_mask_rcnn_r50_fpn/
│      └─ voc_sparse_rcnn_r50_fpn/
└─ Resnet
    ├─ data
    │   └─ caltech-101/
    │       ├─ 101_ObjectCategories/
    │       └─ Annotations/
    ├─ runs
    │   └─ caltech101/       ← TensorBoard logs & checkpoints
    └─ splits/
        ├─ train.txt
        ├─ val.txt
        └─ test.txt
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

- **ResNet-18**: best test accuracy ~87.3 % (see `Resnet/runs/caltech101/`).
- **Mask R-CNN**: bbox mAP ≈ 45 %, segm mAP ≈ 45 % (see `work_dirs/voc_mask_rcnn_r50_fpn/`).
- **Sparse R-CNN**: bbox mAP ≈ 70 %, (see `work_dirs/voc_sparse_rcnn_r50_fpn/`).

Model Checkpoints can be downloaded at 