# On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

This repository is the official implementation of [On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](https://arxiv.org/abs/2410.01405).

## Setup

```shell
conda env create -f environment.yml
conda activate timestep
```

## Experiments

### Dynamic Programming

```bash
cd CoT_benchmark

# Generate the data
python3 tasks/ED/data.py --file "/data/ED/60_24" --length 60 --train_size 1e6 --test_size 1e3 --using 24 # Here `using` + 2 = the max size of working vocabulary.
python3 tasks/LIS/data.py --file 
python3 tasks/CYK/data.py --file 

# Train the model
num_loop=100
model_name="LoopedGPT" # TimeDependentLoopedGPT
task="ED" # LIS, CYK

torchrun --standalone --nproc_per_node=2 train.py --file "data/${task}" --folder "tasks/${task}" --output_dir "output/${task}/${model_name}_${num_loop}" \
 --wandb_name "ED_60_${model_name}_${num_loop}" --model "$model_name" --maxlen 127 --maxdata 127 --vocab 211 --num_range 180 --learning_rate 1e-4 \
 --weight_decay 0.01 --drop 0.0 --batch_size 64 --epoch 100 --warmup 5 --dmodel 256 --head 4 --num_layer 1 --num_loop "$num_loop"
```

### Language Modeling: WikiText-103

```bash
cd nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/train_looped.py
torchrun --standalone --nproc_per_node=2 train.py config/train_time_dependent.py
torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py
```
You can control parameters in `config/train_looped.py` and `config/train_time_dependent.py`.

### Machine Translation: WMT16 Denmark-English

```bash
cd seq2seq

# Download and prepare the data
bash scripts/downloaders/wmt16_en_de.sh

# Train the model
bash scripts/train/train.sh

# Evaluate
bash scripts/eval/eval_bleu.sh

```

## Acknowledgement

- [Code for "Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective"](https://github.com/guyuntian/CoT_benchmark)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Seq2Seq in PyTorch](https://github.com/eladhoffer/seq2seq.pytorch)