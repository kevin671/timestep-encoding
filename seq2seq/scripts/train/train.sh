#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -g gk36
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

conda activate timestep

export WANDB_CONFIG_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_DATA_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_CACHE_DIR="/work/gg45/g45004/timestep-encoding/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"./data/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

WARMUP="4000"
LR0="512**(-0.5)"
NUM_GPUS=4

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} main.py \
  --save transformer_base \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model Transformer \
  --model-config "{'num_layers': 6, 'hidden_size': 512, 'num_heads': 8, 'inner_linear': 2048, 'dropout':0.1}" \
  --data-config "{'moses_pretok':True,'tokenization':'bpe', 'tokenization_config':{'num_symbols':32000}, 'shared_vocab':True}" \
  --b 64 \
  --chunk-batch 2 \
  --max-tokens 6400 \
  --keep-checkpoints 10 \
  --eval-batch-size 16 \
  --label-smoothing 0.1 \
  --trainer Seq2SeqTrainer \
  --optimization-config "[{'step_lambda':
                          \"lambda t: { \
                              'optimizer': 'Adam', \
                              'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
                              'betas': (0.9, 0.98), 'eps':1e-9}\"
                          }]"