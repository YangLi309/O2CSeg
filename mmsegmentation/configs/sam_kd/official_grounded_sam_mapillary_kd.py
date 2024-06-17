_base_ = [
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


data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[122.7709, 116.7460, 104.0937],
        std=[68.5005, 66.6322, 70.3232],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=1024,
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'
)

model = dict(
    type='GroundedDinoSAM',
    exp_name='groundedsam_mapillary',
    dataset='mapillary',
    data_preprocessor=data_preprocessor,
    dino_cfg='configs/sam_kd/official_GroundingDINO_SwinB.py',
    dino_checkpoint='checkpoints/groundingdino_swinb_cogcoor.pth',
    sam_cfg=dict(
        sam_encoder_version='vit_h'
    ),
    sam_checkpoint='checkpoints/sam_hq_vit_h.pth',
    output_classes=[classname.lower() for classname in metainfo['classes']],
    bbox_threshold=0.2,
    sam_threshold=0.5,
    class_refinement=False,
    visualize_results=False,
    color_palette=metainfo['palette']
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(keep_ratio=True, scale=(
    #     2048,
    #     1024,
    # ), type='Resize'),
    # have to change the size because segment anything only accepts images with a maximum edge length of 1024 pixels
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=test_pipeline), sampler=dict(shuffle=False))
test_dataloader = val_dataloader
