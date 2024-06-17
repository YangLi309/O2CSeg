_base_ = [
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)

metainfo = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])


data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[122.7709, 116.7460, 104.0937],
        std=[68.5005, 66.6322, 70.3232],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=640,
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'
)

model = dict(
    type='GroundedDinoSAM',
    exp_name='groundedsam_voc',
    dataset='cityscapes',
    data_preprocessor=data_preprocessor,
    dino_cfg='configs/sam_kd/official_GroundingDINO_SwinB.py',
    dino_checkpoint='checkpoints/groundingdino_swinb_cogcoor.pth',
    sam_cfg=dict(
        sam_encoder_version='vit_h'
    ),
    sam_checkpoint='checkpoints/sam_hq_vit_h.pth',
    output_classes=metainfo['classes'],
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
