_base_ = [
    '../_base_/datasets/cityscapes_segdet_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py', './grounding_dino_swin_b.py'
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
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
)

class_mapping = dict(
        vegetation=['bush', 'tree', 'plants'],
        rider=['rider', 'motorcyclist', 'biker', 'cyclist'],
        wall=['individual wall', 'wall'],
        terrain=['terrain', 'grass', 'sand', 'soil']
)

class_refined_idx2output_idx = dict()
class_output_idx2refined_idx = dict()
refined_classes = list()
output_classes = metainfo['classes']

for output_class in output_classes:
    output_class_id = output_classes.index(output_class)
    if output_class in class_mapping:
        refined_classes.extend(class_mapping[output_class])
        for new_class in class_mapping[output_class]:
            new_class_id = refined_classes.index(new_class)
            class_refined_idx2output_idx[new_class_id] = output_class_id
            if output_class_id not in class_output_idx2refined_idx:
                class_output_idx2refined_idx[output_class_id] = list()
            class_output_idx2refined_idx[output_class_id].append(new_class_id)
    else:
        refined_classes.append(output_class)
        new_class_id = refined_classes.index(output_class)
        class_refined_idx2output_idx[new_class_id] = output_class_id
        class_output_idx2refined_idx[output_class_id] = [new_class_id]

metainfo['refined_classes'] = tuple(refined_classes)

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

seg_data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[73.15835921071147, 82.90891754262579, 72.39239876194173],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=1024,
        std=[47.675755341815076, 48.494214368814504, 47.73654632544151],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'
)

det_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[73.15835921071147, 82.90891754262579, 72.39239876194173],
        pad_mask=False,
        std=[47.675755341815076, 48.494214368814504, 47.73654632544151],
        type='mmdet.DetDataPreprocessor'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape',
    #                                       'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label','text',
    #                                       'caption_prompt', 'custom_entities'))
    dict(type='PackSegInputs')
]

model = dict(
    type='GroundedDinoSAM',
    data_preprocessor=data_preprocessor,
    det_data_preprocessor=det_preprocessor,
    seg_data_preprocessor=seg_data_preprocessor,
    dino_cfg=dict(
        cfg_path='../mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_cityscapes_kd.py',
        bbox_threshold=0.1
    ),
    dino_checkpoint='checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth',
    sam_cfg=dict(
        sam_encoder_version='vit_h',
        batch_size=4,
    ),
    sam_checkpoint='../Grounded-Segment-Anything/checkpoints/sam_hq_vit_h.pth',
    output_classes=output_classes,
    refined_classes=refined_classes,
    class_refined_idx2output_idx=class_refined_idx2output_idx,
    class_output_idx2refined_idx=class_output_idx2refined_idx
)

train_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = val_dataloader
