_base_ = [
    '../_base_/models/san_vit-b16.py',
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
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

class_mapping = {
        'vegetation': ['bush', 'tree', 'plants'],
        'pole': ['pillar', 'spindle', 'post'],
        "traffic sign": ['traffic sign', 'bollard', 'delineator'],
        'rider': ['rider', 'biker', 'cyclist'],
        'wall': ['wall', 'individual wall', 'standing wall', 'independent wall', 'isolated wall'],
        'terrain': ['terrain', 'grass', 'sand', 'soil'],
        'train': ['train', 'subway', 'tram', 'light rail'],
}

model_path = './checkpoints/san-vit-l14_20230907-a11e098f.pth'

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

num_classes = len(metainfo['classes'])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

albu_tta_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(
                    type='Albu',
                    transforms=albu_tta_transforms,
                ),
                dict(
                    type='Albu',
                    transforms=albu_tta_transforms,
                ),
                dict(
                    type='Albu',
                    transforms=albu_tta_transforms,
                ),
                dict(
                    type='Albu',
                    transforms=albu_tta_transforms,
                )
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]


# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2, dataset=dict(metainfo=metainfo))
val_dataloader = dict(
    batch_size=1, dataset=dict(metainfo=metainfo, pipeline=test_pipeline))
test_dataloader = val_dataloader
# test_dataloader = dict(
#     dataset=dict(
#         data_prefix=dict(img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
#         metainfo=metainfo, pipeline=test_pipeline),
#         sampler=dict(shuffle=False)
# )


data_preprocessor = dict(
    mean=[73.15835921071147, 82.90891754262579, 72.39239876194173],
    std=[47.675755341815076, 48.494214368814504, 47.73654632544151],
    size_divisor=1024,
    test_cfg=dict(size_divisor=32)
)

load_from = 'checkpoints/san-vit-l14_20230907-a11e098f.pth'

model = dict(
    type='MultimodalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/ViT-L-14-336px.pth',
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
        vocabulary=refined_classes,
        # dataset_name='cityscapes',
        type='LearnablePromptCLIPTextEncoder',
        cache_feature=False,
        num_contexts=4,
        is_class_specific=False,
        is_context_prompt_learnable=True,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        cat_bg=True,
        output_dims=768,
    ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        # num_classes=len(metainfo['classes']),
        exp_name='san_learnable_prompt_non_csc_cityscapes_conf',
        dataset='cityscapes',
        color_palette=metainfo['palette'],
        num_classes=len(refined_classes),
        refined_classes=refined_classes,
        refine_class_mapping=class_mapping,
        class_refinement=True,
        class_refined_idx2output_idx=class_refined_idx2output_idx,
        class_output_idx2refined_idx=class_output_idx2refined_idx,
        output_classes=metainfo['classes'],
        san_cfg=dict(clip_channels=1024, cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        ),
        kd_training=False,
        visualize_results=False,
        visualize_classes=['wall'],
        vis_dir='./vis_dir/san_learnable_prompt_non_csc_cityscapes_conf/',

        crop_size = crop_size,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=1.0,
                class_weight=[1.0] * len(metainfo['classes']) + [0.1]
            )
        ]
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
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

param_scheduler = [
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
