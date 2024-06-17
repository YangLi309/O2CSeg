_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

num_gpus = 8
max_iters = int(160000/num_gpus)
val_interval = int(16000/num_gpus)

crop_size = (1024, 1024)
data_preprocessor = dict(
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

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

checkpoint = 'checkpoints/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa
num_classes = len(metainfo['classes'])
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=crop_size,
        embed_dims=192,
        num_layers=12,
        num_heads=3),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=192,
        channels=192,
        num_heads=3,
        embed_dims=192,
        num_classes=num_classes,
        kd_training=True,
        crop_size = crop_size,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.0
        )
    ),
)

optimizer = dict(lr=0.001, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=3)
val_dataloader = dict(batch_size=1)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=max_iters,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),

]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='auto', max_keep_ckpts=5)
)
