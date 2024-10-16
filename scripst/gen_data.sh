#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gu15
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

root_dir="/work/gg45/g45004/parallel-looped-tf"

# Arithmetic Expression  ###########################################

NUMBER_OF_OPERATORS=16
DATA_DIR="${root_dir}/data/arithmetic_expression/${NUMBER_OF_OPERATORS}"

#python3 tasks/arithmetic/data.py \
#    --file ${DATA_DIR} \
#    --length ${NUMBER_OF_OPERATORS} \
#    --train_size 1e6 \
#    --test_size 1e3\
#    --number_range 11\

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
    
LEN_OF_FIRST_STRING=100
DATA_DIR="${root_dir}/data/ED/${LEN_OF_FIRST_STRING}"

python3 tasks/ED/data.py \
    --file ${DATA_DIR} \
    --length ${LEN_OF_FIRST_STRING} \
    --train_size 1e5 \
    --test_size 1e2\
    --using 11
