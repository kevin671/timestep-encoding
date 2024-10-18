#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -g gu14
#PJM -j
#PJM --fs /work

export WANDB_CONFIG_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_DATA_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_CACHE_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

source /work/gg45/g45004/.bashrc

ROOT_DIR="/work/gg45/g45004/timestep-encoding" 

# Arithmetic Expression  ###########################################
COMPLEXITY=20
DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression/"${COMPLEXITY}
TASK=${ROOT_DIR}"/tasks/arithmetic"
OUTPUT_DIR=${ROOT_DIR}"/output/arithmetic"
MAXLEN=$((4 * COMPLEXITY + 3))
MAXDATA=${MAXLEN}
VOCAB_SIZE=21
NUM_RANGE=11

# Linear Equation  ###########################################
#COMPLEXITY=3
#DATA_DIR="${ROOT_DIR}/data/linear_equation/"${COMPLEXITY}
#TASK="${ROOT_DIR}/tasks/equation"
#MAXLEN=46
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=22
#NUM_RANGE=11

# Edit Distance  ###########################################
#COMPLEXITY=60
#LEN_OF_FIRST_STRING=60
#DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
#TASK=${ROOT_DIR}"/tasks/ED"
#MAXLEN=127
#MAXDATA=${MAXLEN}
#NUM_RANGE=180
#VOCAB_SIZE=211

MODEL="HyperLoopedGPT" # LoopedGPT, HyperLoopedGPT
LAYER=1
LOOP=50

OUTPUT_DIR=${ROOT_DIR}"/output/$(basename "$TASK")_"${COMPLEXITY}"/"${MODEL}"_"${LOOP}
WANDB_NAME="$(basename "$TASK")_"${COMPLEXITY}"_"${MODEL}"_"${LOOP}

torchrun --standalone --nproc_per_node=2 train.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --output_dir ${OUTPUT_DIR}\
 --wandb_name ${WANDB_NAME}\
 --model ${MODEL}\
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
 --model_path "/work/gg45/g45004/timestep-encoding/output/arithmetic_20/HyperLoopedGPT_50/epneao3k/epoch_10.pt"