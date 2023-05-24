
dataset_type = 'CocoDataset'
data_root = r'E:\Jiang\data\MOT\SkySat\bbox_centernet\\'
classes = ('car', )
img_norm_cfg = dict(
    mean=[
        123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28,
        103.53
    ],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[
            123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28,
            103.53
        ],
        std=[
            58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375
        ],
        to_rgb=False),
    dict(type='Pad', size=(256, 256)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[
            123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28,
            103.53
        ],
        std=[
            58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375
        ],
        to_rgb=False),
    dict(type='Pad', size=(256, 256)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
    dict(type='WrapFieldsToLists')
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'pseudo.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'gt.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'gt.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
        )
evaluation = dict(interval=1, metric='proposal_fast', iou_thrs=[0.3])

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='RPN',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=9,  # todo
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[4, 6, -1, -1, -1]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[2],
            strides=[4, 6, -1, -1, -1]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.07, 0.07, 0.11, 0.11]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
            center_ratio=0.2,
            ignore_ratio=0.5)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.3),
            min_bbox_size=0)))
work_dir = './work_dirs\ga_rpn_r50_fpn_1x8_1x_skysat'
gpu_ids = range(0, 1)
