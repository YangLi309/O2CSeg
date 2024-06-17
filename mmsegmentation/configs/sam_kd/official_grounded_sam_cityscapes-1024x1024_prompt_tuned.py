_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)
exp_name = "grounded_sam_cityscapes_prompt_tuned_train_terrain_rider"
metainfo = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [241, 161, 177], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

class_mapping = {
        'road': ['highway', 'street', 'avenue', 'drive way'],
        'vegetation': ['vegetation', 'bush', 'tree', 'plants'],
        # 'pole': ['pole', 'pillar', 'spindle', 'post'],
        "traffic sign": ['traffic sign', 'bollard', 'delineator'],
        'rider': ['rider', 'motorcyclist', 'biker', 'cyclist'],
        'wall': ['wall', 'individual wall', 'standing wall', 'independent wall', 'isolated wall', 'single wall', 'partition wall'],
        'terrain': ['terrain', 'grass', 'sand', 'soil'],
        'train': ['subway', 'tram', 'light rail'],
        'sidewalk': ['sidewalk', 'pavement', 'pedestrian path', 'pedestrian way']
        # sidewalk=dict(
        #     prompts=['sidewalk', 'pavement', 'footpath', 'pedestrian path', 'pedestrian way', 'footway', 'street', 'road'],
        #     mask=[1, 1, 1, 1, 1, 1, 0, 0]
        # )
}

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
    exp_name=exp_name,
    data_preprocessor=data_preprocessor,
    dino_cfg='configs/sam_kd/official_GroundingDINO_SwinB.py',
    dino_checkpoint='checkpoints/groundingdino_swinb_cogcoor.pth',
    sam_cfg=dict(
        sam_encoder_version='vit_h'
    ),
    sam_checkpoint='checkpoints/sam_hq_vit_h.pth',
    bbox_threshold=0.2,
    sam_threshold=0.5,
    color_palette=metainfo['palette'],
    refine_class_mapping=class_mapping,
    output_classes=output_classes,
    refined_classes=refined_classes,
    class_refined_idx2output_idx=class_refined_idx2output_idx,
    class_output_idx2refined_idx=class_output_idx2refined_idx,
    class_refinement=True,
    visualize_results=False,
    visualize_classes=['train', 'pole', 'terrain', 'rider', 'person'],
    kd_training=False
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
val_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=test_pipeline), sampler=dict(shuffle=True))
# test_dataloader = dict(
#     dataset=dict(
#         data_prefix=dict(img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
#         metainfo=metainfo, pipeline=test_pipeline),
#         sampler=dict(shuffle=False)
# )

test_dataloader = val_dataloader
