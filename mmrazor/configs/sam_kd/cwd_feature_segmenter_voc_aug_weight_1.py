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
        student_recorders=dict(
            # enc3=dict(type='ModuleOutputs', source='backbone.layers.3'),
            # enc7=dict(type='ModuleOutputs', source='backbone.layers.7'),
            # enc11=dict(type='ModuleOutputs', source='backbone.layers.11'),
            enc_last_layer=dict(type='ModuleOutputs', source='decode_head.enc_patch_emb_module'),
            # dec_emb=dict(type='ModuleInputs', source='decode_head.dec_patch_emb_module'),

        ),
        teacher_recorders=dict(
            # enc3=dict(type='ModuleOutputs', source='backbone.layers.7'),
            # enc7=dict(type='ModuleOutputs', source='backbone.layers.15'),
            # enc11=dict(type='ModuleOutputs', source='backbone.layers.23'),
            enc_last_layer=dict(type='ModuleOutputs', source='decode_head.enc_patch_emb_module'),
            # dec_emb=dict(type='ModuleInputs', source='decode_head.dec_patch_emb_module'),
        ),
        connectors=dict(
            enc_last_layer_connector=dict(
                type='ConvModuleConnector',
                in_channel=192,
                out_channel=1024),
        ),
        distill_losses=dict(
            loss_cwd_last_layer_enc=dict(type='ChannelWiseDivergence', tau=1, loss_weight=1),
        ),
        loss_forward_mappings=dict(
            loss_cwd_last_layer_enc=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='enc_last_layer',
                    connector='enc_last_layer_connector'),
                preds_T=dict(from_student=False, recorder='enc_last_layer')),
        )
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
