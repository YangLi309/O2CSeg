import torch

checkpoint_path = '/home/aslab/code/yang_code/semantic_segmentation/mmsegmentation/checkpoints/best_san_non_csc_4ctx_cityscapes.pth'
output_checkpoint_path = '/home/aslab/code/yang_code/semantic_segmentation/mmsegmentation/checkpoints/best_san_learnable_prompt_non_csc_4ctx_cityscapes.pth'
checkpoint = torch.load(checkpoint_path)
new_checkpoint = {
    'meta': checkpoint['meta'],
    'message_hub': checkpoint['message_hub']
}
state_dict = {}
for key, value in checkpoint['state_dict'].items():
    if 'architecture' in key:
        new_key = key.replace('architecture.', '')
        state_dict[new_key] = value
new_checkpoint['state_dict'] = state_dict
torch.save(new_checkpoint, output_checkpoint_path)
