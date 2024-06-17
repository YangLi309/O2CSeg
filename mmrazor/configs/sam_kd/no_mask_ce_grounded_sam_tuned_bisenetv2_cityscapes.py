_base_ = [
    'mmseg::_base_/datasets/cityscapes_1024x1024.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py'
]
batch_size = 1
num_gpus = 1
# num_gpus = 8
max_iters = int(160000/num_gpus)
val_interval = int(16000/num_gpus)

# resume training
# resume = True
# load_from = '/home/li309/yang_seg/exp_results/kd_logits_san_segmenter_voc_w_projection/best_miou884.pth'

teacher_ckpt = 'checkpoints/san-vit-l14_20230907-a11e098f.pth'
teacher_cfg_path = 'mmseg::sam_kd/official_grounded_sam_cityscapes-1024x1024_prompt_tuned_kd.py'
student_cfg_path = 'mmseg::sam_kd/bisenetv2_fcn_4xb8-160k_cityscapes-1024x1024_kd.py'

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
    dataset=dict(metainfo=metainfo)
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
    teacher_ckpt=None,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_ce=dict(type='CrossEntropyLoss', loss_weight=batch_size),
            # loss_kd_after_proj=dict(type='KLDivergence', tau=1, loss_weight=batch_size, reduction='mean')
        ),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.mask_result_module'),
            # logits_w_projection=dict(type='ModuleOutputs', source='decode_head.mask_after_projection_result_module')
        ),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='mask_result_module'),
            loss_mask=dict(type='ModuleOutputs', source='background_mask_result_module'),
        ),
            # logits=dict(type='ModuleOutputs', source='background_mask_result_module')),
        loss_forward_mappings=dict(
            loss_ce=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                # loss_mask=dict(from_student=False, recorder='loss_mask')
            ),
            # loss_kd_after_proj=dict(
            #     preds_S=dict(from_student=True, recorder='logits_w_projection'),
            #     preds_T=dict(from_student=False, recorder='logits'))
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
    checkpoint=dict(type='CheckpointHook', save_best='auto', interval=val_interval, max_keep_ckpts=5)
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0),
    clip_grad=None)

auto_scale_lr = dict(enable=True, base_batch_size=num_gpus * batch_size)
