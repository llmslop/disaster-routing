#!/bin/env python3

import sys
from typing import IO

import pandas as pd
from excel_utils import dump_excel


def pivot_results(input: IO[bytes], output: IO[bytes]):
    with pd.ExcelWriter(output) as out:
        df = pd.read_excel(input, "raw")
        topologies = ["nsfnet", "cost239", "usb"]
        sizes = [10, 20, 50, 100, 200, 300, 400]

        # scenario 1
        columns = ["Req. count"]
        for topo in topologies:
            columns.append(f"HCDP-{topo}")
            columns.append(f"Flow-{topo}")
            columns.append(f"%imp-{topo}")

        scen1 = pd.DataFrame(columns=columns)

        for size in sizes:
            result: dict[str, object] = {"Req. count": size}
            for topo in topologies:
                sub = df[df["instance"].str.contains(f"{topo}-{size:04}")]
                pivot = sub.pivot(index="instance", columns="solver", values="score")

                result[f"HCDP-{topo}"] = pivot["greedy+npm+ls"].mean()
                result[f"Flow-{topo}"] = (
                    pivot[["flow+npm+ls", "flow_dp+npm+ls"]].min(axis=1).mean()
                )
                result[f"%imp-{topo}"] = (
                    (result[f"HCDP-{topo}"] - result[f"Flow-{topo}"])
                    / result[f"HCDP-{topo}"]
                    * 100
                )
                result[f"%imp-{topo}"] = f"{result[f'%imp-{topo}']:.2f}%"
                result[f"HCDP-{topo}"] = f"{result[f'HCDP-{topo}']:.4f}"
                result[f"Flow-{topo}"] = f"{result[f'Flow-{topo}']:.4f}"

            newstuff = pd.DataFrame([result])
            scen1 = pd.concat([scen1, newstuff], ignore_index=True)

        dump_excel(scen1, out, "scenario1", index=False)

        # scenario 2
        columns = ["Req. count"]
        for topo in topologies:
            columns.append(f"GA+PRI-{topo}")
            columns.append(f"GA+HI-{topo}")
            columns.append(f"%imp-{topo}")

        scen2 = pd.DataFrame(columns=columns)

        for size in sizes:
            result: dict[str, object] = {"Req. count": size}
            for topo in topologies:
                sub = df[df["instance"].str.contains(f"{topo}-{size:04}")]
                pivot = sub.pivot(index="instance", columns="solver", values="score")

                result[f"GA+PRI-{topo}"] = pivot["sga_ri(fpga, npm)"].mean()
                result[f"GA+HI-{topo}"] = pivot["sga_hi(fpga, npm)"].mean()
                result[f"%imp-{topo}"] = (
                    (result[f"GA+PRI-{topo}"] - result[f"GA+HI-{topo}"])
                    / result[f"GA+PRI-{topo}"]
                    * 100
                )
                result[f"%imp-{topo}"] = f"{result[f'%imp-{topo}']:.2f}%"
                result[f"GA+HI-{topo}"] = f"{result[f'GA+HI-{topo}']:.4f}"
                result[f"GA+PRI-{topo}"] = f"{result[f'GA+PRI-{topo}']:.4f}"

            newstuff = pd.DataFrame([result])
            scen2 = pd.concat([scen2, newstuff], ignore_index=True)

        dump_excel(scen2, out, "scenario2", index=False)

        # scenario 3
        columns = ["Req. count"]
        for topo in topologies:
            columns.append(f"Flow-{topo}")
            columns.append(f"GA+HI-{topo}")
            columns.append(f"%imp-{topo}")

        scen3 = pd.DataFrame(columns=columns)

        for size in sizes:
            result: dict[str, object] = {"Req. count": size}
            for topo in topologies:
                sub = df[df["instance"].str.contains(f"{topo}-{size:04}")]
                pivot = sub.pivot(index="instance", columns="solver", values="score")

                result[f"Flow-{topo}"] = (
                    pivot[["flow+npm+ls", "flow_dp+npm+ls"]].min(axis=1).mean()
                )
                result[f"GA+HI-{topo}"] = pivot["sga_hi(fpga, npm)"].mean()
                result[f"%imp-{topo}"] = (
                    (result[f"Flow-{topo}"] - result[f"GA+HI-{topo}"])
                    / result[f"Flow-{topo}"]
                    * 100
                )
                result[f"%imp-{topo}"] = f"{result[f'%imp-{topo}']:.2f}%"
                result[f"Flow-{topo}"] = f"{result[f'Flow-{topo}']:.4f}"
                result[f"GA+HI-{topo}"] = f"{result[f'GA+HI-{topo}']:.4f}"

            newstuff = pd.DataFrame([result])
            scen3 = pd.concat([scen3, newstuff], ignore_index=True)

        dump_excel(scen3, out, "scenario3", index=False)

        # scenario 4
        columns = ["Req. count"]
        for topo in topologies:
            columns.append(f"HCDP-{topo}")
            columns.append(f"GA+HI-{topo}")
            columns.append(f"%imp-{topo}")

        scen4 = pd.DataFrame(columns=columns)

        for size in sizes:
            result: dict[str, object] = {"Req. count": size}
            for topo in topologies:
                sub = df[df["instance"].str.contains(f"{topo}-{size:04}")]
                pivot = sub.pivot(index="instance", columns="solver", values="score")

                result[f"HCDP-{topo}"] = pivot["greedy+npm+ls"].mean()
                result[f"GA+HI-{topo}"] = pivot["sga_hi(fpga, npm)"].mean()
                result[f"%imp-{topo}"] = (
                    (result[f"HCDP-{topo}"] - result[f"GA+HI-{topo}"])
                    / result[f"HCDP-{topo}"]
                    * 100
                )
                result[f"%imp-{topo}"] = f"{result[f'%imp-{topo}']:.2f}%"
                result[f"HCDP-{topo}"] = f"{result[f'HCDP-{topo}']:.4f}"
                result[f"GA+HI-{topo}"] = f"{result[f'GA+HI-{topo}']:.4f}"

            newstuff = pd.DataFrame([result])
            scen4 = pd.concat([scen4, newstuff], ignore_index=True)

        dump_excel(scen4, out, "scenario4", index=False)


if __name__ == "__main__":
    pivot_results(sys.stdin.buffer, sys.stdout.buffer)
