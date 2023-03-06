# The new config inherits a base config to highlight the necessary modification
# _base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
_base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py'
# _base_ = '/root/autodl-nas/code/ODIS/CarDD_detection/configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(create_symlink=False)
classes = ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')
n_classes = len(classes)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=n_classes),
        mask_head=dict(num_classes=n_classes)))

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
# load_from = '/root/autodl-nas/model/pretrained/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth'
load_from = '/root/autodl-nas/model/pretrained/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth'
# load_from = '/root/autodl-nas/model/pretrained/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth'
