_base_ = [
    '../_base_/models/retinanet_r50_fpn_skysat.py',
    '../_base_/datasets/skysat_detection.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
