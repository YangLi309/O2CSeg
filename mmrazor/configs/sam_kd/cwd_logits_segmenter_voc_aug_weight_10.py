_base_ = [
    'mmseg::_base_/datasets/pascal_voc12_aug.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py'
]
num_gpus = 8

teacher_ckpt = '/home/li309/yang_seg/segmentation/mmsegmentation/checkpoints/segmenter_vit-l_mask_8xb1-160k_voc_aug_512x512.pth'
teacher_cfg_path = 'mmseg::segmenter/segmenter_vit-l_mask_8xb1-160k_voc_aug_512x512.py'  # noqa: E501
student_cfg_path = 'mmseg::segmenter/segmenter_vit-t_mask_8xb1-160k_voc_aug_512x512.py'

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
            loss_cwd=dict(type='ChannelWiseDivergence', tau=1, loss_weight=10)),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))
    )
)

find_unused_parameters = True

train_dataloader = dict(batch_size=12)

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
