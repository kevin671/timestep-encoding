import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import aggregate_metrics, eval_looped_model, eval_unlooped_model

fig_hparam = {
    "figsize": (8, 5),
    "labelsize": 28,
    "ticksize": 20,
    "linewidth": 5,
    "fontsize": 15,
    "titlesize": 20,
    "markersize": 15,
}

# font specification
fontdict = {
    "family": "serif",
    "size": fig_hparam["fontsize"],
}

device = torch.device("cuda:0")


def get_model(model, result_dir, run_id, step, best=False):
    if best:
        model_path = os.path.join(result_dir, run_id, "model_best.pt")
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        best_err = torch.load(model_path, map_location="cpu")["loss"]
        print("saved model with loss:", best_err)
    if step == -1:
        model_path = os.path.join(result_dir, run_id, "state.pt")
        state_dict = torch.load(model_path, map_location="cpu")["model_state_dict"]
    else:
        model_path = os.path.join(result_dir, run_id, "model_{}.pt".format(step))
        state_dict = torch.load(model_path, map_location="cpu")["model"]

    #     return state_dict
    # unwanted_prefix = "_orig_mod."
    # for k, v in list(state_dict.items()):
    #    if k.startswith(unwanted_prefix):
    #        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    return model


from tasks import DecisionTree

sample_size = 1280
batch_size = 128
n_points = 101
n_dims_truncated = 20
n_dims = 20

real_task = DecisionTree(sample_size, n_points, n_dims, n_dims_truncated, device)
xs, ys = real_task.xs, real_task.ys

result_errs = {}
result_loop_errs = {}

from models import (
    TransformerModel,
    TransformerModelLooped,
    TransformerModelTimeDependentLooped,
)

result_dir = "/work/gg45/g45004/timestep-encoding/in_context/results2/decision_tree_baseline"

run_id = "1118123336-DT_baseline-0bce"

n_positions = 101
n_embd = 256
n_layer = 12
n_head = 8

step = 200000

model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head)
model = get_model(model, result_dir, run_id, step)
model = model.to(device)
err, loop_err = eval_unlooped_model(model, xs, ys)

result_errs[run_id] = err

"""
# run_id = "0721152049-LR_loop_L1_ends{30}_T{15}-0668"
run_ids = [
    "1117030200-DT_loop_L1_ends{70}_T{15}-482b",
    "1117151547-DT_time_L1_ends{70}_T{15}-bfc1",
    "1118085010-DT_loop_L1_ends{12}_T{12}-86ba",
    "1118003007-DT_time_L1_ends{12}_T{12}-3784",
]

n_positions = 101
n_embd = 256
n_head = 8
# T = 70
n_layer = 1

# model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)
step = 200000

for run_id in run_ids:
    if "time" in run_id:
        model = TransformerModelTimeDependentLooped(n_dims, n_positions, n_embd, n_layer, n_head)
    else:
        model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)

    model = get_model(model, result_dir, run_id, step)
    model = model.to(device)

    # get T from run id
    T = int(run_id.split("T{")[1].split("}")[0])

    err, loop_err = eval_looped_model(model, xs, ys, loop_max=T)

    result_errs[run_id] = err
    result_loop_errs[run_id] = loop_err
    print(loop_err)
"""

result_errs_agg = aggregate_metrics(result_errs, n_dims_truncated)

print(result_errs_agg.keys())

import matplotlib

fig, ax = plt.subplots(1, figsize=fig_hparam["figsize"])

err_result_dict_agg = result_errs_agg

cmap = matplotlib.cm.get_cmap("coolwarm")
# result_name_list = run_ids
result_name_list = [run_id]
colors = cmap(np.linspace(0, 1, len(result_name_list)))
for idx, model_name in enumerate(result_name_list):
    err = err_result_dict_agg[model_name]["mean"]
    ax.plot(
        err,
        color=colors[idx],
        lw=fig_hparam["linewidth"],
        label=model_name.capitalize(),
    )
    low = err_result_dict_agg[model_name]["bootstrap_low"]
    high = err_result_dict_agg[model_name]["bootstrap_high"]
    ax.fill_between(range(len(low)), low, high, alpha=0.3, color=colors[idx])

    # save err as text
    np.savetxt(
        "/work/gg45/g45004/timestep-encoding/in_context/results2/DT_err_{}.csv".format(model_name),
        err,
        delimiter=",",
    )

ax.tick_params(axis="both", labelsize=fig_hparam["ticksize"])
ax.axhline(1, color="k", ls="--", lw=fig_hparam["linewidth"])
# ax.set_ylim(-0.1, 1.25)
# Y軸のスケールを対数に設定
ax.set_yscale("log")
ax.set_ylim(0.001, 0.1)

# plt.xticks(np.arange(0, n_points))
plt.rc("font", family="serif")
ax.set_xlabel("in-context examples", fontsize=fig_hparam["labelsize"])
y_label = ax.set_ylabel("squared error", fontsize=fig_hparam["labelsize"])
legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fig_hparam["fontsize"])

plt.savefig(
    "/work/gg45/g45004/timestep-encoding/in_context/results2/Figures/DT_err.png",
    dpi=600,
    bbox_inches="tight",
)
plt.close()
