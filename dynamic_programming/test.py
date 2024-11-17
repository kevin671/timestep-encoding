import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import set_seed

from algorithmic_task.model import GPT, TimeDependentLoopedGPT, LoopedGPT

parser = argparse.ArgumentParser(description="test")

parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--file", type=str, default="Data")
parser.add_argument("--folder", type=str, default="arithmetic")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--chain", action="store_true", default=False)
parser.add_argument("--rpe", action="store_true", default=False)
parser.add_argument("--maxlen", type=int, default=120)
parser.add_argument("--maxdata", type=int, default=120)
parser.add_argument("--maxans", type=int, default=30)
parser.add_argument("--vocab", type=int, default=21)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--drop", type=float, default=0.1)
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--num_loop", type=int, default=100)
parser.add_argument("--head", type=int, default=4)
parser.add_argument("--num_range", type=int, default=11)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--parallel", action="store_true", default=False)

args = parser.parse_args()

print(args)

import sys

sys.path.append(args.folder)

from dataloader import getLoader  # type: ignore

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
main_process = 0
set_seed(2023)
dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.barrier()
model = LoopedGPT(args).cuda()
# model = GPT(args).cuda()
# model = TimeDependentLoopedGPT(args).cuda()
if args.model_path:
    model.load_state_dict(torch.load(args.model_path), strict=True)
model = DDP(model, device_ids=[local_rank])
model_without_ddp = model.module
# _, test_loader = getLoader(args)
train_loader, test_loader = getLoader(args)

import itertools


def evaluate(cur_loader):
    Sum, correct = 0, 0
    cur_loader.sampler.set_epoch(0)
    for _, (input_ids, y, _) in enumerate(itertools.islice(cur_loader, 100)):
        inputs, y = input_ids.cuda(), y.cuda()
        logits = model(inputs)
        Sum += torch.as_tensor(inputs.shape[0]).cuda()
        truth = torch.where(y > 0, 1, 0)
        predict = torch.where(torch.argmax(logits, dim=2) == y, 1, 0) * truth
        correct += torch.sum(
            torch.where(torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0)
        )
    dist.all_reduce(correct)
    dist.all_reduce(Sum)
    return correct / Sum


model.eval()
with torch.no_grad():
    acc = evaluate(train_loader)
    if dist.get_rank() == main_process:
        print(f"Train acc:{acc}")


model.eval()
with torch.no_grad():
    acc = evaluate(test_loader)
    if dist.get_rank() == main_process:
        print(f"Test acc:{acc}")
