_base_ = [
    '/root/autodl-tmp/mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '/root/autodl-tmp/mmdetection/configs/_base_/datasets/coco_instance.py',
    '/root/autodl-tmp/mmdetection/configs/schedule_1x.py',
    '/root/autodl-tmp/mmdetection/configs/_base_/default_runtime.py'
]

class_names = (
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike',
    'person','pottedplant','sheep','sofa',
    'train','tvmonitor'
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(class_names)),
        mask_head=dict(num_classes=len(class_names))
    )
)

data_root = 'data/VOCdevkit/'

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        ann_file='annotations/voc0712_train.json',
        data_prefix=dict(img=''),              # ← 关键修改
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        ann_file='annotations/voc0712_val.json',
        data_prefix=dict(img=''),              # ← 关键修改
        test_mode=True,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/voc0712_val.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator
