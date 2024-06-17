_base_ = [
    'mmseg::_base_/datasets/pascal_voc12_aug.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py'
]
num_gpus = 1
batch_size = 2
teacher_ckpt = 'checkpoints/san-vit-l14_20230907-a11e098f.pth'
teacher_cfg_path = 'mmseg::sam_kd/san-vit-l14_voc12aug-512x512_kd.py'  # noqa: E501
student_cfg_path = 'mmseg::sam_kd/san-vit-l14_voc12aug-512x512_learnable_prompt_non_csc_4ctx_kd.py'
val_interval = 300
max_iters = 60000
metainfo = dict(
    classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    palette=[[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
             [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
             [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
             [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])

dataset_train = _base_.dataset_train
dataset_train['metainfo'] = metainfo
dataset_aug = _base_.dataset_aug
dataset_aug['metainfo'] = metainfo

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(datasets=[dataset_train, dataset_aug]),
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
                         class_logits_threshold_path='./checkpoints/voc_class_san_threshold_logits_0.8.pkl',
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
