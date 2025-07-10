#!/bin/env python3

from typing import IO
import pandas as pd
import sys
import jq


def dump_excel(
    df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str, index: bool = False
):
    df.to_excel(writer, sheet_name=sheet_name, index=index)
    start_loc = 0
    if index:
        column_length = max(len(str(c)) for c in df.index)
        if isinstance(df.index.name, str):
            column_length = max(column_length, len(df.index.name))
        writer.sheets[sheet_name].set_column(start_loc, start_loc, column_length)
        start_loc += 1
    for i, column in enumerate(df):
        column_length = df[column].astype(str).map(len).max()
        if isinstance(column, tuple):
            column_length = max(column_length, max(len(c) for c in column))
        else:
            column_length = max(column_length, len(str(column)))
        # add some more space
        column_length = max(10, min(column_length + 5, column_length * 1.1))
        writer.sheets[sheet_name].set_column(
            start_loc + i, start_loc + i, column_length
        )


def pivot_results(input: IO[str], output: IO[bytes]):
    with pd.ExcelWriter(output) as out:
        results = (
            jq.compile(
                """
                .[] | {
                  router:.config.router._short_,
                  instance: .config.instance.path,
                  mofi: first(.run_log[] | select(.msg="Final solution") | .args.mofi),
                  total_fs: first(.run_log[] | select(.msg="Final solution") | .args.total_fs),
                }
                """
            )
            .input_text(input.read())
            .all()
        )
        df = pd.DataFrame.from_dict(results)
        dump_excel(df, out, "raw")
        df = df.pivot_table(["mofi", "total_fs"], "instance", "router")
        dump_excel(df, out, "pivot", index=True)


if __name__ == "__main__":
    pivot_results(sys.stdin, sys.stdout.buffer)
