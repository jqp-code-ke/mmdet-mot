_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_skysat.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        in_channels=9,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        # 一般在 Caffe 模式下，requires_grad=False，也就是说 ResNet 的所有 BN 层参数都不更新并且全局均值和方差也不再改变，
        # 而在 PyTorch 模式下，除了 frozen_stages 的 BN 参数不更新外，其余层 BN 参数还是会更新的。
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
    ))
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 梯度均衡参数
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180])
runner = dict(type='EpochBasedRunner', max_epochs=210)
