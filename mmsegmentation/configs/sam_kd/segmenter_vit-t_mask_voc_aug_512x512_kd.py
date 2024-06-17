_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'

data_preprocessor = dict(
    size=crop_size,
    mean=[115.40426167733553, 110.06103779748479, 101.88298592833684],
    std=[70.56859777572389, 69.65910509081601, 72.7374357413688]
)
checkpoint = 'checkpoints/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

metainfo = dict(
    classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    palette=[[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
             [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
             [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
             [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        embed_dims=192,
        num_layers=12,
        num_heads=3),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=192,
        channels=192,
        num_classes=len(metainfo['classes']),
        num_heads=3,
        embed_dims=192,
        kd_training=True,
        crop_size = crop_size,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.0
        )
    ),
)

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
dataset_train = _base_.dataset_train
dataset_train['metainfo'] = metainfo
dataset_aug = _base_.dataset_aug
dataset_aug['metainfo'] = metainfo
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=4,
    dataset=dict(datasets=[dataset_train, dataset_aug])
)
val_dataloader = dict(batch_size=1, dataset=dict(metainfo=metainfo))
test_dataloader = val_dataloader
