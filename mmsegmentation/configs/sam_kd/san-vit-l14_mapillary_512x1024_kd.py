_base_=[
    '../_base_/models/san_vit-b16.py',
    '../_base_/datasets/mapillary_v1_65.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size=(512, 1024)

metainfo=dict(
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

num_classes=len(metainfo['classes'])

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader=dict(batch_size=1)
val_dataloader=dict(batch_size=1)
test_dataloader=val_dataloader

data_preprocessor=dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=512,
    test_cfg=dict(size_divisor=32))
model=dict(
    type='MultimodalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/ViT-L-14-336px.pth',
    encoder_resolution=0.7,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17),
    ),
    text_encoder=dict(
        vocabulary=metainfo['classes'],
        type='CLIPTextEncoder',
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        output_dims=768,
    ),
    decode_head=dict(
        exp_name='san_vit_l14_mapillary',
        color_palette=metainfo['palette'],
        dataset='mapillary',
        output_classes=metainfo['classes'],
        type='SideAdapterCLIPHead',
        num_classes=len(metainfo['classes']),
        san_cfg=dict(clip_channels=1024, cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        ),
        kd_training=True,
        crop_size=crop_size,
        # visualize_results=True,
        # visualize_classes=['Bird', 'Ground Animal', 'Curb'],
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                class_weight=[1.0 for _ in range(len(metainfo['classes']))] + [0.1]
            )]
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper=dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler=[
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
