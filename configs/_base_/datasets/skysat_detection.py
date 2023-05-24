# dataset settings
dataset_type = 'CocoDataset'
data_root = r'E:\Jiang\data\MOT\SkySat\bbox_centernet\\'
classes = ('car',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28, 103.53],  # [0.40789654, 0.44719302, 0.47026115]
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375],  # [0.28863828, 0.27408164, 0.27809835]
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
    dict(type='WrapFieldsToLists')
]
data = dict(
    samples_per_gpu=8,  # 单个 GPU 的 Batch size
    workers_per_gpu=4,  # 单个 GPU 分配的数据加载线程数
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
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='bbox', iou_thrs=[0.3])
