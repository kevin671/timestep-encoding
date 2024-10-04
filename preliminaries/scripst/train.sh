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


# Arithmetic Expression  ###########################################
#DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression"
#DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression/100K"
#TASK=${ROOT_DIR}"/tasks/arithmetic"
#OUTPUT_DIR=${ROOT_DIR}"/output/arithmetic"
#MAXLEN=23
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=21
#NUM_RANGE=11

#NUMBER_OF_OPERATORS=10
#DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression/"${NUMBER_OF_OPERATORS}
#TASK=${ROOT_DIR}"/tasks/arithmetic"
#OUTPUT_DIR=${ROOT_DIR}"/output/arithmetic/"${NUMBER_OF_OPERATORS}"/looped"
#MAXLEN=43
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=21
#NUM_RANGE=11

#NUMBER_OF_OPERATORS=20
#DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression/"${NUMBER_OF_OPERATORS}
#TASK=${ROOT_DIR}"/tasks/arithmetic"
#MAXLEN=83
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=21
#NUM_RANGE=11

#NUMBER_OF_OPERATORS=16
#DATA_DIR=${ROOT_DIR}"/data/arithmetic_expression/"${NUMBER_OF_OPERATORS}
#TASK=${ROOT_DIR}"/tasks/arithmetic"
#MAXLEN=67
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=21
#NUM_RANGE=11

# Edit Distance  ###########################################
#DATA_DIR=${ROOT_DIR}"/data/ED"
#TASK=${ROOT_DIR}"/ED"
#OUTPUT_DIR=${ROOT_DIR}"/output/ED"
#MAXLEN=47
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=91
#NUM_RANGE=60

#LEN_OF_FIRST_STRING=32
#DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
#TASK=${ROOT_DIR}"/tasks/ED"
#OUTPUT_DIR=${ROOT_DIR}"/output/ED/"${LEN_OF_FIRST_STRING}
#MAXLEN=87
#MAXDATA=${MAXLEN}
#VOCAB_SIZE=120
#NUM_RANGE=90

#LEN_OF_FIRST_STRING=40
#DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
#TASK=${ROOT_DIR}"/tasks/ED"
#OUTPUT_DIR=${ROOT_DIR}"/output/ED/"${LEN_OF_FIRST_STRING}
#MAXLEN=87
#MAXDATA=${MAXLEN}
#NUM_RANGE=120
#VOCAB_SIZE=151

#LEN_OF_FIRST_STRING=60
#DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
#DATA_DIR=${ROOT_DIR}"/data/ED/"${LEN_OF_FIRST_STRING}
#TASK=${ROOT_DIR}"/tasks/ED"
#MAXLEN=127
#MAXDATA=${MAXLEN}
#NUM_RANGE=180
#VOCAB_SIZE=211

# Linear Equation  ###########################################
DATA_DIR="${ROOT_DIR}/data/linear_equation"
TASK="${ROOT_DIR}/tasks/equation"
OUTPUT_DIR="${ROOT_DIR}/output/equation"
MAXLEN=46
MAXDATA=${MAXLEN}
VOCAB_SIZE=22
NUM_RANGE=11

LAYER=1
LOOP=100

# OUTPUT_DIR=${ROOT_DIR}"/output/arithmetic_"${NUMBER_OF_OPERATORS}"_Loop_"${LOOP}
# OUTPUT_DIR=${ROOT_DIR}"/output/ED_"${LEN_OF_FIRST_STRING}"_Loop_"${LOOP}
OUTPUT_DIR=${ROOT_DIR}"/output/LinEq_Looped_"${LOOP}
#WANDB_NAME="Arithmetic_"${NUMBER_OF_OPERATORS}"_Loop_"${LOOP}
WANDB_NAME="ED_"${LEN_OF_FIRST_STRING}"_Hyper_"${LOOP}

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
 --batch_size 128\
 --epoch 100\
 --warmup 5\
 --dmodel 256\
 --head 4\
 --num_layer ${LAYER}\
 --num_loop ${LOOP}\