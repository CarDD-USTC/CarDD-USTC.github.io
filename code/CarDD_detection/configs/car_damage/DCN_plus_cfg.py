# Author: wxk
# Time: 2022/11/15 15:43

# The new config inherits a base config to highlight the necessary modification
_base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py'

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
classes = ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')
n_classes = len(classes)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(True, True, True, True)),
    roi_head=dict(
        bbox_head=dict(num_classes=n_classes,
                       loss_cls=dict(
                           type='FocalLoss',
                           use_sigmoid=True,
                           gamma=2.0,
                           alpha=0.5,
                           loss_weight=1.0)
                       ),
        mask_head=dict(num_classes=n_classes)))

# Modify dataset related settings
dataset_type = 'COCODataset'
data = dict(
    samples_per_gpu=4,  # batch size
    workers_per_gpu=2,
    train=dict(
        pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 1200)],
                    multiscale_mode='range',
                    keep_ratio=False,
                    backend='pillow'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])],
        img_prefix='/root/autodl-fs/data/CarDD_COCO/train2017/',
        classes=classes,
        ann_file='/root/autodl-fs/data/CarDD_COCO/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/root/autodl-fs/data/CarDD_COCO/val2017/',
        classes=classes,
        ann_file='/root/autodl-fs/data/CarDD_COCO/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/root/autodl-fs/data/CarDD_COCO/test2017/',
        classes=classes,
        ann_file='/root/autodl-fs/data/CarDD_COCO/annotations/instances_test2017.json'))

optimizer = dict(lr=0.005)  # LR
evaluation = dict(interval=6)
checkpoint_config = dict(create_symlink=False, interval=6)
log_config = dict(interval=500)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/root/autodl-nas/model/pretrained/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth'
