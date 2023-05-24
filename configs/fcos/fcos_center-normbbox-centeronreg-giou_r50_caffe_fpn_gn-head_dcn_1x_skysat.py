_base_ = 'fcos_r50_caffe_fpn_gn-head_1x8_1x_skysat.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.3)))

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675, 103.530, 116.280, 123.675, 103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(256, 256)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'E:/jqp/data/COCO/SkySat/'
classes = ('car',)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
optimizer_config = dict(_delete_=True, grad_clip=None)

lr_config = dict(warmup='linear')
