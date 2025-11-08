#!/bin/env python3

from argparse import ArgumentParser
from sys import stdin, stdout
from typing import IO

import jq
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from excel_utils import dump_excel


def dump_data(input: IO[str], output: IO[bytes]):
    results = (
        jq.compile(
            """
        .[] | {
          solver: first(.run_log[] | select(.msg=="Solver details") | .args.name),
          instance: .config.instance.path,
          fitness: [.run_log[] | select(.msg=="Best individual") | .args.fitness],
        }
        """
        )
        .input_text(input.read())
        .all()
    )

    df = pd.DataFrame(results)
    df = df[df["solver"] == "sga_hi(fpga, npm)"]
    df = df.explode("fitness").reset_index(drop=True)
    df["generation"] = df.groupby("instance").cumcount()
    df["size"] = df["instance"].str.extract(r"-(\d+)-")[0].astype(int)

    with pd.ExcelWriter(output) as out:
        dump_excel(df, out, "raw")
    return


def size_num_to_str(size: int) -> str:
    if size == 10:
        return "Small instances"
    elif size == 20 or size == 50:
        return "Medium instances"
    elif size == 100:
        return "Large instances"
    else:
        return "Huge instances"


def plot(input: IO[bytes], savefig: str | None):
    df = pd.read_excel(input, "raw")
    df["size"] = df["size"].apply(size_num_to_str)
    # df = df[df["size"] == "huge"]
    topologies = {
        "nsfnet": "NSFNET",
        "cost239": "COST239",
        "usb": "US Backbone",
    }
    df["topology"] = (
        df["instance"].str.split("/").str[-1].str.split("-").str[0].map(topologies)
    )

    # sns.set_theme(style="whitegrid", palette="deep")
    # sns.set(font_scale=1.75)
    # plt.rcParams.update({"font.size": 14, "font.family": "Times New Roman"})
    sns.set_theme(
        style="whitegrid", palette="deep", font_scale=1.75, font="Times New Roman"
    )

    g = sns.relplot(
        data=df,
        x="generation",
        y="fitness",
        hue="topology",
        hue_order=["NSFNET", "COST239", "US Backbone"],
        kind="line",
        col="size",
        # col_wrap=2,
        col_order=[size_num_to_str(x) for x in [10, 20, 100, 200]],
        estimator="mean",  # average across runs
        errorbar=("ci", 95),  # shaded band
    )

    g.set_titles(template="{col_name}")
    g.set_axis_labels("Instance size", "")
    g.set_ylabels("Fitness")
    g.set_xlabels("Generation")

    g._legend.remove()
    handles, labels = [], []
    for ax in g.axes.flatten():
        h, l = ax.get_legend_handles_labels()  # noqa: E741
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    g.axes[0][-1].legend(handles, labels, loc="lower right")

    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.title("SGA convergence rate")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Scenario 3 data processing")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    dump_parser = subparsers.add_parser("dump", help="Dump data from JSON to Excel")
    plot_parser = subparsers.add_parser("plot", help="Plot data from Excel")
    plot_parser.add_argument(
        "--savefig", type=str, help="Path to save the figure (optional)"
    )

    args = parser.parse_args()

    if args.mode == "dump":
        dump_data(stdin, stdout.buffer)
    elif args.mode == "plot":
        plot(stdin.buffer, args.savefig)
