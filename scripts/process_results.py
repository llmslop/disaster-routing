#!/bin/env python3

import sys
from typing import IO

import jq
import pandas as pd
from excel_utils import dump_excel


def pivot_results(input: IO[str], output: IO[bytes]):
    with pd.ExcelWriter(output) as out:
        results = (
            jq.compile(
                """
            .[] | {
              solver: first(.run_log[] | select(.msg=="Solver details") | .args.name),
              instance: .config.instance.path,
              mofi: first(.run_log[] | select(.msg=="Final solution") | .args.mofi),
              total_fs: first(.run_log[] | select(.msg=="Final solution")
                                         | .args.total_fs),
              score: first(.run_log[] | select(.msg=="Final solution") | .args.score),
              time: last(.run_log[]).time_unix - first(.run_log[]).time_unix,
            }
            """
            )
            .input_text(input.read())
            .all()
        )
        df = pd.DataFrame.from_dict(results)
        dump_excel(df, out, "raw")
        df = df.pivot_table(
            values=["mofi", "total_fs", "score", "time"],
            index="instance",
            columns="solver",
            margins_name="Avg.",
            margins=True,
        )
        dump_excel(df, out, "pivot", index=True)


if __name__ == "__main__":
    pivot_results(sys.stdin, sys.stdout.buffer)
