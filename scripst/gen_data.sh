#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gk36
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

root_dir="/work/gg45/g45004/timestep-encoding"

# Arithmetic Expression  ###########################################

NUMBER_OF_OPERATORS=16
DATA_DIR="${root_dir}/data/arithmetic_expression/${NUMBER_OF_OPERATORS}"

#python3 tasks/arithmetic/data.py \
#    --file ${DATA_DIR} \
#    --length ${NUMBER_OF_OPERATORS} \
#    --train_size 1e6 \
#    --test_size 1e3\
#    --number_range 11\

# Linear Equation  ###########################################
DATA_DIR="${root_dir}/data/linear_equation"
NUMBER_OF_VARIABLES=12

#python3 tasks/equation/data.py \
#    --file ${DATA_DIR} \
#    --length ${NUMBER_OF_VARIABLES} \
#    --train_size 1e6 \
#    --test_size 1e5\
#    --number_range 11


# Edit Distance  ###########################################
DATA_DIR="${root_dir}/data/ED"
LEN_OF_FIRST_STRING=24

#python3 tasks/ED/data.py \
#    --file ${DATA_DIR} \
#    --length ${LEN_OF_FIRST_STRING} \
#    --train_size 1e6 \
#    --test_size 1e5\
#    --using 8

#LEN_OF_FIRST_STRING=32
#DATA_DIR="${root_dir}/data/ED/${LEN_OF_FIRST_STRING}"

#python3 tasks/ED/data.py \
#    --file ${DATA_DIR} \
#    --length ${LEN_OF_FIRST_STRING} \
#    --train_size 1e8 \
#    --test_size 1e5\
#    --using 8

#LEN_OF_FIRST_STRING=40
#DATA_DIR="${root_dir}/data/ED/${LEN_OF_FIRST_STRING}"

#python3 tasks/ED/data.py \
#    --file ${DATA_DIR} \
#    --length ${LEN_OF_FIRST_STRING} \
#    --train_size 1e6 \
#    --test_size 1e5\
#    --using 8


#LEN_OF_FIRST_STRING=60
#ALPHABET=24
#DATA_DIR="${root_dir}/data/ED/${LEN_OF_FIRST_STRING}_${ALPHABET}"

#python3 tasks/ED/data.py \
#    --file ${DATA_DIR} \
#    --length ${LEN_OF_FIRST_STRING} \
#    --train_size 1e6 \
#    --test_size 1e3\
#    --using ${ALPHABET}
    
#LEN_OF_FIRST_STRING=100
#DATA_DIR="${root_dir}/data/ED/${LEN_OF_FIRST_STRING}"

#python3 tasks/ED/data.py \
#    --file ${DATA_DIR} \
#    --length ${LEN_OF_FIRST_STRING} \
#    --train_size 1e5 \
#    --test_size 1e2\
#    --using 11

# Longest Increasing Subsequence  ###########################################
DATA_DIR="${root_dir}/data/LIS"

LEN_INPUTS=100
NUM_RANGE=250

python3 tasks/LIS/data.py \
    --file ${DATA_DIR}/${LEN_INPUTS} \
    --length ${LEN_INPUTS} \
    --train_size 1e6 \
    --test_size 1e3\
    --number_range ${NUM_RANGE}