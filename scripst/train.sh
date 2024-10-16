#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gu14
#PJM -j
#PJM --fs /work

export WANDB_CONFIG_DIR="/work/gg45/g45004/timestep-encoding/preliminaries/tmp"
export WANDB_DATA_DIR="/work/gg45/g45004/timestep-encoding/preliminaries/tmp"
export WANDB_CACHE_DIR="/work/gg45/g45004/timestep-encoding/preliminaries/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

source /work/gg45/g45004/.bashrc

ROOT_DIR="/work/gg45/g45004/timestep-encoding/preliminaries" 


LEN_OF_FIRST_STRING=60
DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
TASK=${ROOT_DIR}"/tasks/ED"
MAXLEN=127
MAXDATA=${MAXLEN}
NUM_RANGE=180
VOCAB_SIZE=211

LAYER=3
LOOP=33

# OUTPUT_DIR=${ROOT_DIR}"/output/arithmetic_"${NUMBER_OF_OPERATORS}"_Loop_"${LOOP}
OUTPUT_DIR=${ROOT_DIR}"/output/ED_"${LEN_OF_FIRST_STRING}"_Loop_"${LOOP}
#WANDB_NAME="Arithmetic_"${NUMBER_OF_OPERATORS}"_Loop_"${LOOP}
WANDB_NAME="ED_"${LEN_OF_FIRST_STRING}"_Loop_"${LOOP}

torchrun --standalone --nproc_per_node=1 train.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --output_dir ${OUTPUT_DIR}\
 --wandb_name ${WANDB_NAME}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --weight_decay 0.01\
 --learning_rate 1e-4\
 --drop 0.0\
 --batch_size 64\
 --epoch 100\
 --warmup 5\
 --dmodel 256\
 --head 4\
 --num_layer ${LAYER}\
 --num_loop ${LOOP}\