#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gk36
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TRITON_CACHE_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_CONFIG_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

torchrun --standalone --nproc_per_node=1 train.py config/train_looped.py
