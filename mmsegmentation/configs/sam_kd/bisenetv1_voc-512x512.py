_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
data_root = '../data/voc/VOC2012/'
num_gpus = 8
batch_size = 8
max_iters = int(320000/num_gpus)
val_interval = int(16000/num_gpus)
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

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    mean=[115.40426167733553, 110.06103779748479, 101.88298592833684],
    std=[70.56859777572389, 69.65910509081601, 72.7374357413688]
)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=len(metainfo['classes']),
    ),
)
_base_.model['auxiliary_head'][0]['num_classes'] = len(metainfo['classes'])
_base_.model['auxiliary_head'][1]['num_classes'] = len(metainfo['classes'])
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=300),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=300,
        end=max_iters,
        by_epoch=False,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=batch_size, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='auto', max_keep_ckpts=5)
)

auto_scale_lr = dict(enable=True, base_batch_size= num_gpus * batch_size)