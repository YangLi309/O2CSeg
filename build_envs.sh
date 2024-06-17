#!/bin/bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

#conda create -n O2CSeg python=3.10
#conda activate O2CSeg
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ftfy wandb optuna seaborn --quiet
pip install openmim --quiet
mim install -y mmengine
mim install -y "mmcv>=2.0.0rc4, <2.2.0."
cd mmsegmentation
pip install -e .
cd ..
cd mmrazor
pip install -e .
cd ..
cd Grounded-Segment-Anything
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch] --quiet

