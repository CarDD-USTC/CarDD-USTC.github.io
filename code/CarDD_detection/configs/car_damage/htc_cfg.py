# The new config inherits a base config to highlight the necessary modification
# _base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/htc/htc_r50_fpn_1x_coco.py'
# _base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/htc/htc_r50_fpn_20e_coco.py'
_base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/htc/htc_r101_fpn_20e_coco.py'
# _base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/htc/htc_without_semantic_r50_fpn_1x_coco.py'

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(create_symlink=False)
classes = ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')
n_classes = len(classes)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=n_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=n_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=n_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=n_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=n_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=n_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]))

# Modify dataset related settings
dataset_type = 'COCODataset'
data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=2,
    train=dict(
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
optimizer = dict(lr=0.0025)  # LR
# evaluation = dict(interval=4) 


# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = '/root/autodl-nas/model/pretrained/htc_r50_fpn_1x_coco_20200317-7332cf16.pth'
# load_from = '/root/autodl-nas/model/pretrained/htc_r50_fpn_20e_coco_20200319-fe28c577.pth'
load_from = '/root/autodl-nas/model/pretrained/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth'
