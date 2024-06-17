_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/pascal_context_59.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
data_root = './data/voc/VOC2010/'
batch_size = 8
num_gpus = 1
max_iters = int(160000/num_gpus)
val_interval = int(16000/num_gpus)

crop_size = (480, 480)
data_preprocessor = dict(
    size=crop_size,
    mean=[116.62302698701093, 111.51832432295127, 103.04587280997863],
    std=[70.08495346581704, 69.23702390734218, 72.62436811695048],
)

metainfo = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

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
        crop_size=crop_size,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0
        )
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(480, 480))
)

optimizer = dict(lr=0.001, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=batch_size, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=max_iters,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),

]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='auto', max_keep_ckpts=5)
)
auto_scale_lr = dict(enable=True, base_batch_size= num_gpus * batch_size)
