_base_ = [
    '../_base_/datasets/skysat_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=9,
        norm_eval=False,
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=False),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    to_rgb=False)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(
        type='RandomCenterCropPad',
        ratios=None,
        border=None,
        mean=[0, 0, 0, 0, 0, 0, 0, 0, 0],
        std=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        to_rgb=False,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect',
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'border'),
         keys=['img']),
    dict(type='WrapFieldsToLists')
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
