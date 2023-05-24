_base_ = './retinanet_r50_fpn_1x_sd.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        in_channels= 9,
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
    ))
# learning policy
lr_config = dict(step=[8*4, 11*4])
runner = dict(type='EpochBasedRunner', max_epochs=12*4)  # 最好参数