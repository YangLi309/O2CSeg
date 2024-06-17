_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
data_root = './data/voc/VOC2010/'
num_gpus = 1
batch_size = 16
max_iters = int(160000/num_gpus)
val_interval = int(16000/num_gpus)
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

crop_size = (480, 480)
data_preprocessor = dict(
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    mean=[116.62302698701093, 111.51832432295127, 103.04587280997863],
    std=[70.08495346581704, 69.23702390734218, 72.62436811695048],
    size_divisor=480,
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor'
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
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=batch_size, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=5, dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='auto', max_keep_ckpts=5)
)
auto_scale_lr = dict(enable=True, base_batch_size= num_gpus * batch_size)