_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
metainfo = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    mean=[115.40426167733553, 110.06103779748479, 101.88298592833684],
    std=[70.56859777572389, 69.65910509081601, 72.7374357413688]
)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=len(metainfo['classes'])
    ),
)
_base_.model.auxiliary_head[0]['num_classes'] = len(metainfo['classes'])
_base_.model.auxiliary_head[1]['num_classes'] = len(metainfo['classes'])
_base_.model.auxiliary_head[2]['num_classes'] = len(metainfo['classes'])

dataset_train = _base_.dataset_train
dataset_train['metainfo'] = metainfo
dataset_aug = _base_.dataset_aug
dataset_aug['metainfo'] = metainfo

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
train_dataloader = dict(batch_size=12, num_workers=4, dataset=dict(datasets=[dataset_train, dataset_aug]))
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
