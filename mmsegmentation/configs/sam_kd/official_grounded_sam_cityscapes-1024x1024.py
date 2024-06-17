_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)

metainfo = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[73.15835921071147, 82.90891754262579, 72.39239876194173],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=1024,
        std=[47.675755341815076, 48.494214368814504, 47.73654632544151],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'
)

model = dict(
    type='GroundedDinoSAM',
    dataset='cityscapes',
    data_preprocessor=data_preprocessor,
    dino_cfg='configs/sam_kd/official_GroundingDINO_SwinB.py',
    dino_checkpoint='checkpoints/groundingdino_swinb_cogcoor.pth',
    sam_cfg=dict(
        sam_encoder_version='vit_h'
    ),
    sam_checkpoint='checkpoints/sam_hq_vit_h.pth',
    output_classes=metainfo['classes'],
    text_threshold=0.25,
    bbox_threshold=0.2,
    sam_threshold=0.5,
    class_refinement=False,
    visualize_results=True,
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
