#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -g gu14
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
NUM_GPUS=2

NUM_LOOPS=3
MODEL_TYPE=LoopedTransformer  # TimeDependentLoopedTransformer # LoopedTransformer

python -m torch.distributed.launch --standalone --nproc_per_node=${NUM_GPUS} main.py \
  --save ${MODEL_TYPE}_${NUM_LOOPS} \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model ${MODEL_TYPE} \
  --model-config "{'num_layers': 1, 'num_loops': ${NUM_LOOPS}, 'hidden_size': 1024, 'num_heads': 16, 'inner_linear': 4096, 'dropout':0.1, 'prenormalized': True}" \
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