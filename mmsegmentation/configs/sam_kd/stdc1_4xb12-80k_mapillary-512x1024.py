_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/mapillary_v1_65.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

data_root = './data/mapillary/'
batch_size = 8
num_gpus = 1
max_iters = int(160000/num_gpus)
val_interval = int(16000/num_gpus)

crop_size = (512, 1024)
data_preprocessor = dict(
    size=crop_size,
    mean=[106.84911361612228, 116.91568184164608, 119.82839073449622],
    std=[67.33273446882501, 70.18045856618774, 77.27218287827769],
)

metainfo = dict(
    classes=('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
             'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
             'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane',
             'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist',
             'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
             'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
             'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
             'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant',
             'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
             'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
             'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)',
             'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',
             'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
             'Wheeled Slow', 'Car Mount', 'Ego Vehicle'),
    palette=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
             [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
             [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
             [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
             [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
             [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
             [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
             [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
             [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
             [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
             [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
             [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
             [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
             [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
             [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
             [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10]])

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=len(metainfo['classes']),
        kd_training=True
    ),
)
_base_.model.auxiliary_head[0]['num_classes'] = len(metainfo['classes'])
_base_.model.auxiliary_head[1]['num_classes'] = len(metainfo['classes'])

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=max_iters,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=batch_size, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='mIoU', max_keep_ckpts=5)
)
auto_scale_lr = dict(enable=True, base_batch_size= num_gpus * batch_size)
