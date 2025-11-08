#!/bin/env python3

from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1, font="Times New Roman")

# Example data ------------------------
instance_sizes = [10, 20, 50, 100, 200, 300, 400]  # replace with your sizes
algorithms = {
    "sga_ri(fpga, npm)": "GA+PRI",
    "sga_hi(fpga, npm)": "GA+HI",
}
topologies = {
    "nsfnet": "NSFNET",
    "cost239": "COST239",
    "usb": "US Backbone",
}


df = pd.read_excel(stdin.buffer, "raw")
df["topology"] = (
    df["instance"].str.split("/").str[-1].str.split("-").str[0].map(topologies)
)
df["instance_size"] = df["instance"].str.extract(r"-(\d+)-")[0].astype(int)

df = df.melt(
    id_vars=["solver", "topology", "instance_size"],
    value_vars=["time", "score"],
    var_name="metric",
    value_name="value",
)
df = df[df["solver"].isin(list(algorithms.keys()))]
df["solver"] = df["solver"].map(algorithms)

g = sns.catplot(
    data=df,
    x="instance_size",
    y="value",
    hue="solver",
    col="topology",
    row="metric",
    kind="bar",
    sharex=True,
    sharey="row",
    height=3,
    aspect=1.2,
)

# Adjust scales
g.set_titles(template="{col_name}")
for ax in g.axes[-1, :]:  # last row, all columns
    ax.set_title("")
g.set_axis_labels("Instance size", "")
for (row_val, col_val), ax in g.axes_dict.items():
    if row_val == "time":
        ax.set_yscale("log")
        ax.set_ylabel("Runtime (s, log scale)")
    elif row_val == "score":
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Objective value")
        ax.set_ylim(0.5, 1.3)

g._legend.remove()
handles, labels = [], []
for ax in g.axes.flatten():
    h, l = ax.get_legend_handles_labels()  # noqa: E741
    for hh, ll in zip(h, l):
        if ll not in labels:
            handles.append(hh)
            labels.append(ll)
g.axes[0, 0].legend(handles, labels, loc="upper left")
# for topo in topologies:
#     runtime.append([[], []])
#     result.append([[], []])
#     for size in instance_sizes:
#         sub = df[df["instance"].str.contains(f"{topo}-{size:04}")]
#         pivot = sub.pivot(index="instance", columns="solver", values="score")
#         hi_score = pivot["sga_hi(fpga, npm)"].mean()
#         ri_score = pivot["sga_ri(fpga, npm)"].mean()
#         pivot = sub.pivot(index="instance", columns="solver", values="time")
#         hi_time = pivot["sga_hi(fpga, npm)"].mean()
#         ri_time = pivot["sga_ri(fpga, npm)"].mean()
#
#         runtime[-1][0].append(ri_time)
#         runtime[-1][1].append(hi_time)
#
#         result[-1][0].append(ri_score)
#         result[-1][1].append(hi_score)
#
# fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
#
# for col, topo in enumerate(topologies):
#     # Runtime row (log scale)
#     ax = axes[0, col]
#     ax.bar(x - bar_width / 2, runtime[col][0], width=bar_width, label=algorithms[0])
#     ax.bar(x + bar_width / 2, runtime[col][1], width=bar_width, label=algorithms[1])
#     ax.set_yscale("log")
#     ax.set_title(topologies[topo])
#     ax.set_xticks(x)
#     ax.set_xticklabels(instance_sizes)
#     if col == 0:
#         ax.set_ylabel("Runtime (s, log scale)")
#         ax.legend()
#
#     # Result row
#     ax = axes[1, col]
#     ax.bar(x - bar_width / 2, result[col][0], width=bar_width, label=algorithms[0])
#     ax.bar(x + bar_width / 2, result[col][1], width=bar_width, label=algorithms[1])
#     ax.set_xlabel("Instance size")
#     ax.axhline(1, color="black", linestyle="--", linewidth=1)  # dotted reference line
#     if col == 0:
#         ax.set_ylabel("Objective value")
#     ax.set_xticks(x)
#     ax.set_xticklabels(instance_sizes)
#
#
plt.tight_layout()
if len(argv) > 1:
    plt.savefig(argv[1])
plt.show()
