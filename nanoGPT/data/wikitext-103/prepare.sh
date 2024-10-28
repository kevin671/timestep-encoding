#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM --fs /work

export HF_DATASETS_CACHE="/work/gg45/g45004/timestep-encoding/"

source /work/gg45/g45004/.bashrc

python data/wikitext-103/prepare.py