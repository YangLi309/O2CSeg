# Activation Boundaries Loss (ABLoss)

> [Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/pdf/1811.03233.pdf)

<!-- [ALGORITHM] -->

## Abstract

An activation boundary for a neuron refers to a separating hyperplane that determines whether the neuron is activated or deactivated. It has been long considered in neural networks that the activations of neurons, rather than their exact output values, play the most important role in forming classification friendly partitions of the hidden feature space. However, as far as we know, this aspect of neural networks has not been considered in the literature of knowledge transfer. In this pa- per, we propose a knowledge transfer method via distillation of activation boundaries formed by hidden neurons. For the distillation, we propose an activation transfer loss that has the minimum value when the boundaries generated by the stu- dent coincide with those by the teacher. Since the activation transfer loss is not differentiable, we design a piecewise differentiable loss approximating the activation transfer loss. By the proposed method, the student learns a separating bound- ary between activation region and deactivation region formed by each neuron in the teacher. Through the experiments in various aspects of knowledge transfer, it is verified that the proposed method outperforms the current state-of-the-art [link](https://github.com/bhheo/AB_distillation)

<img width="1184" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187422794-d681ed58-293a-4d9e-9e5b-9937289136a7.png">

## Results and models

### Classification

|      Location       | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                                   Config                                   | Download                                                                                                                                                                                                                                                                                                                                                                                                     |
| :-----------------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backbone (pretrain) | ImageNet | [resnet50](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) |       | 76.55  | 69.90  | [pretrain_config](./abloss_pretrain_backbone_resnet50_resnet18_8xb32_in1k) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/ABLoss/abloss_pretrain_backbone_resnet50_resnet18_8xb32_in1k_20220830_165724-a6284e9f.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/ABLoss/abloss_pretrain_backbone_resnet50_resnet18_8xb32_in1k_20220830_165724-a6284e9f.json) |
|   logits (train)    | ImageNet | [resnet50](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 69.94 | 76.55  | 69.90  |      [train_config](./abloss_logits_resnet50_resnet18_8xb32_in1k.py)       | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/ABLoss/abloss_logits_resnet50_resnet18_8xb32_in1k_20220830_202129-f35edde8.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/ABLoss/abloss_logits_resnet50_resnet18_8xb32_in1k_20220830_202129-f35edde8.json)                       |

## Citation

```latex
@inproceedings{DBLP:conf/aaai/HeoLY019a,
  author    = {Byeongho Heo, Minsik Lee, Sangdoo Yun and Jin Young Choi},
  title     = {Knowledge Transfer via Distillation of Activation Boundaries Formed
               by Hidden Neurons},
  booktitle = {The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI}
               2019, The Thirty-First Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2019, The Ninth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2019, Honolulu, Hawaii,
               USA, January 27 - February 1, 2019},
  pages     = {3779--3787},
  publisher = {{AAAI} Press},
  year      = {2019},
  url       = {https://doi.org/10.1609/aaai.v33i01.33013779},
  doi       = {10.1609/aaai.v33i01.33013779},
  timestamp = {Fri, 07 May 2021 11:57:04 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/HeoLY019a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Getting Started

### Pre-training.

```bash
sh tools/dist_train.sh configs/distill/mmcls/abloss/abloss_pretrain_backbone_resnet50_resnet18_8xb32_in1k.py 8
```

### Modify Distillation training config

open file 'configs/distill/mmcls/abloss/abloss_logits_resnet50_resnet18_8xb32_in1k.py'

```python
# Modify init_cfg in model settings.
# 'pretrain_work_dir' is same as the 'work_dir of pre-training'.
# 'last_epoch' defaults to 'epoch_20' in ABLoss.
init_cfg=dict(
    type='Pretrained', checkpoint='pretrain_work_dir/last_epoch.pth'),
```

### Distillation training.

```bash
sh tools/dist_train.sh configs/distill/mmcls/abloss/abloss_logits_resnet50_resnet18_8xb32_in1k.py 8
```