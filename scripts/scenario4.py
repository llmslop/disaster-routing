#!/bin/env python3

from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"font.size": 14, "font.family": "Times New Roman"})

# Instance sizes and topologies
sizes = [10, 20, 50, 100, 200, 300, 400]
topologies = {
    "nsfnet": "NSFNET",
    "cost239": "COST239",
    "usb": "US Backbone",
}
algorithms = {
    "greedy+npm+ls": "HCDP",
    "sga_hi(fpga, npm)": "GA+HI",
}

df = pd.read_excel(stdin.buffer, "raw")
df["topology"] = (
    df["instance"].str.split("/").str[-1].str.split("-").str[0].map(topologies)
)
df["instance_size"] = df["instance"].str.extract(r"-(\d+)-")[0].astype(int)

df = df[df["solver"].isin(list(algorithms.keys()))]
df = df.groupby(["topology", "instance_size", "solver"], as_index=False)["score"].mean()
df = df.pivot(index=["topology", "instance_size"], columns="solver", values="score")
df["%imp"] = (df["greedy+npm+ls"] - df["sga_hi(fpga, npm)"]) / df["greedy+npm+ls"]
df = df.reset_index()

order = ["US Backbone", "COST239", "NSFNET"]
df["topology"] = pd.Categorical(df["topology"], categories=order, ordered=True)
df = df.sort_values("topology").reset_index(drop=True)

heatmap_data = df.pivot(index="topology", columns="instance_size", values="%imp")

# Optional: annotation text showing solver fitnesses
annot_data = df.pivot(index="topology", columns="instance_size", values="%imp")
annot_data = annot_data.astype(str)
for idx, row in df.iterrows():
    annot_data.at[row["topology"], row["instance_size"]] = (
        f"{row['greedy+npm+ls']:.2f} → {row['sga_hi(fpga, npm)']:.2f}"
        + f"\n(+{row['%imp'] * 100:.2f}%)"
    )

heatmap_data = heatmap_data * 100  # convert to percentage

# Plot (IEEE double-column friendly)
plt.figure(figsize=(8, 4))  # double-column width, reasonable height
g = sns.heatmap(
    heatmap_data.astype(float),
    annot=annot_data,
    fmt="",
    # cmap="RdYlGn",
    center=0,
    cbar_kws={"label": "% improvement"},
    annot_kws={"size": 10},  # readable font for double-column
)
g.set_xlabel("Instance size")
g.set_ylabel("Topology")
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
plt.tight_layout()
if len(argv) > 1:
    plt.savefig(argv[1])
plt.show()
