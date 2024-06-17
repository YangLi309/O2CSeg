_base_ = [
    'mmseg::_base_/datasets/cityscapes_1024x1024.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py'
]
num_gpus = 1
batch_size = 2
teacher_ckpt = 'checkpoints/san-vit-l14_20230907-a11e098f.pth'
teacher_cfg_path = 'mmseg::sam_kd/san-vit-l14_cityscapes_prompt_tuned_kd.py'  # noqa: E501
student_cfg_path = 'mmseg::sam_kd/san-vit-l14_cityscapes_learnable_prompt_kd.py'
val_interval = 300
max_iters = 160000/num_gpus
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

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(metainfo=metainfo),
    num_workers=5
)

val_dataloader = dict(batch_size=1, dataset=dict(metainfo=metainfo), num_workers=5)
test_dataloader = val_dataloader


model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=True),
    teacher=dict(
        cfg_path=teacher_cfg_path,
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    calculate_student_loss=False,
    student_trainable_modules=['text_encoder.context_embeddings'],
    # student_trainable_modules=[],
    student_prompt_learnable=True,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_kd=dict(type='MaskedCrossEntropyLoss',
                         class_logits_threshold_path='./checkpoints/cityscapes_class_san_threshold_logits_0.1.pkl',
                         loss_weight=1
            )
            # loss_kd=dict(type='CrossEntropyLoss', loss_weight=1)
        ),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        loss_forward_mappings=dict(
            loss_kd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')
            )
        )
    )
)

find_unused_parameters = True

# val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
# we dont eval teacher
val_cfg = dict(_delete_=True, type='mmrazor.SelfDistillValLoop')

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='auto', max_keep_ckpts=5)
)

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0)

param_scheduler = [

]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None
)
