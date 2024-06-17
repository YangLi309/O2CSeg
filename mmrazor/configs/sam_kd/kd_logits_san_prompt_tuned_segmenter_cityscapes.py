_base_ = [
    'mmseg::_base_/datasets/pascal_voc12_aug.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py'
]
num_gpus = 1

teacher_ckpt = 'checkpoints/san-vit-l14_20230907-a11e098f.pth'
teacher_cfg_path = 'mmseg::sam_kd/san-vit-l14_voc12aug-512x512_kd.py'  # noqa: E501
student_cfg_path = 'mmseg::sam_kd/segmenter_vit-t_mask_voc_aug_512x512_kd.py'

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
    # num_gpus: 8 -> batch_size: 8
    batch_size=8,
    dataset=dict(datasets=[dataset_train, dataset_aug])
)

val_dataloader = dict(batch_size=1, dataset=dict(metainfo=metainfo))
test_dataloader = val_dataloader

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(
        cfg_path=teacher_cfg_path,
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_kd=dict(type='KLDivergence', tau=1, loss_weight=3, add_projection=True, num_channels=len(metainfo['classes']))),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        loss_forward_mappings=dict(
            loss_kd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'))
        )
    )
)

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=int(1600/num_gpus))

default_hooks = dict(
    checkpoint=dict(interval=int(1600/num_gpus))
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0),
    clip_grad=None)
