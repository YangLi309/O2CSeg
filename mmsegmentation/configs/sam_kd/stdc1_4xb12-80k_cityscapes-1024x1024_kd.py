_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

metainfo = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=len(metainfo['classes']),
        kd_training=True
    ),
)
_base_.model.auxiliary_head[0]['num_classes'] = len(metainfo['classes'])
_base_.model.auxiliary_head[0]['kd_training'] = True
_base_.model.auxiliary_head[1]['num_classes'] = len(metainfo['classes'])
_base_.model.auxiliary_head[1]['kd_training'] = True
_base_.model.auxiliary_head[2]['kd_training'] = True


param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=80000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=12, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
