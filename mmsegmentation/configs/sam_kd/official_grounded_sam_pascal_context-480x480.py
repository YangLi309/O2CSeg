_base_ = [
    '../_base_/datasets/pascal_context_59.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

img_scale = (520, 520)
crop_size = (480, 480)
exp_name = "grounded_sam_pascal_context"
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


data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[116.62302698701093, 111.51832432295127, 103.04587280997863],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=520,
        std=[70.08495346581704, 69.23702390734218, 72.62436811695048],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'
)

model = dict(
    type='GroundedDinoSAM',
    dataset='pascal_context',
    exp_name=exp_name,
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
    dict(keep_ratio=True, scale=(1024, 1024), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=test_pipeline), sampler=dict(shuffle=False))
test_dataloader = val_dataloader
