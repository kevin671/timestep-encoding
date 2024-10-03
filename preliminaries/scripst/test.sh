#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gu14
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

ROOT_DIR="/work/gg45/g45004/parallel-looped-tf" 

LEN_OF_FIRST_STRING=60
DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
TASK=${ROOT_DIR}"/tasks/ED"
MAXLEN=127
MAXDATA=${MAXLEN}
NUM_RANGE=180
VOCAB_SIZE=211


LAYER=1
LOOP=50
MODEL_PATH="/work/gg45/g45004/parallel-looped-tf/output/ED_60_Loop_50/hcz3t8v2/epoch_40.pt"


torchrun --nproc_per_node=1 test.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --batch_size 128\
 --dmodel 256\
 --head 4\
 --num_layer ${LAYER}\
 --num_loop ${LOOP}\
 --model_path ${MODEL_PATH}\

# --parallel \