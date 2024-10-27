# On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

This repository is the official implementation of [On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](https://arxiv.org/abs/2410.01405).

## Setup

```shell
conda env create -f environment.yml
conda activate timestep
```

## Experiments

### Dynamic Programming: Edit Distance

```bash
cd CoT_benchmark

# Generate the data
python3 tasks/ED/data.py --file "/data/ED/60_24" --length 60 --train_size 1e6 --test_size 1e3 --using 24
# Here `using` + 2 = the max size of working vocabulary.

# Train the model
num_loop=100 # 10, 50
model_name="LoopedGPT" # TimeDependentLoopedGPT

torchrun --standalone --nproc_per_node=2 train.py --file "data/ED/60" --folder "tasks/ED" --output_dir "output/ED_60/${model_name}_${num_loop}" \
 --wandb_name "ED_60_${model_name}_${num_loop}" --model "$model_name" --maxlen 127 --maxdata 127 --vocab 211 --num_range 180 --learning_rate 1e-4 \
 --weight_decay 0.01 --drop 0.0 --batch_size 64 --epoch 100 --warmup 5 --dmodel 256 --head 4 --num_layer 1 --num_loop "$num_loop"
```

### Language Modeling: WikiText-103

```bash
cd nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/train_looped.py
torchrun --standalone --nproc_per_node=2 train.py config/train_time_dependent.py
```
You can control parameters in `config/train_looped.py` and `config/train_time_dependent.py`.

### Machine Translation: WMT'14 English to French

Looped Transformers are implemetned in `seq2seq/fairseq/modules/timestep_layer.py` and `seq2seq/fairseq/models/looped_transformer.py`.

```bash
cd seq2seq
pip install --editable ./

# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

# Train the model
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wmt14_en_fr.tokenized.de-en \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# Evaluate
fairseq-generate \
    data-bin/transformer_wmt_en_fr \
    --path checkpoints/transformer_wmt_en_fr/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

## Acknowledgement

- [Code for "Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective"](https://github.com/guyuntian/CoT_benchmark)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [fairseq](https://github.com/facebookresearch/fairseq)