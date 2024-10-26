# On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

We present a Looped Transformers with Timestep Encoding

You can find the paper in [arxiv](https://arxiv.org/abs/2410.01405).

## Setup

```shell
conda env create -f environment.yml
conda activate loop_tf
```

## Experiments

### Dynamic Programming: Edit Distance

```
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
```

### Language Modeling: WikiText-103

```
cd nanoGPT
torchrun --standalone --nproc_per_node=1 train.py config/train_looped.py
```

## Acknowledgement

- [Code for "Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective"](https://github.com/guyuntian/CoT_benchmark/tree/main)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Code for "Lessons on Parameter Sharing across Layers in Transformers"](https://github.com/takase/share_layer_params/tree/main)
