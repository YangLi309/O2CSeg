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

model_path = './checkpoints/san-vit-l14_20230907-a11e098f.pth'

num_classes = len(metainfo['classes'])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
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
        vocabulary=metainfo['classes'],
        # dataset_name='cityscapes',
        type='LearnablePromptCLIPTextEncoder',
        cache_feature=False,
        num_contexts=10,
        is_class_specific=True,
        is_context_prompt_learnable=True,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        cat_bg=True,
        output_dims=768,
    ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        num_classes=len(metainfo['classes']),
        class_refinement=False,
        output_classes=metainfo['classes'],
        san_cfg=dict(clip_channels=1024, cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
        ),
        kd_training=True,
        crop_size=crop_size,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=1.0,
                class_weight=[1.0 for _ in range(len(metainfo['classes']))]
            )]
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
