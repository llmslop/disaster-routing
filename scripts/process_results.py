#!/bin/env python3

from typing import IO
import pandas as pd
import sys
import jq


def is_float(cell: str | float) -> bool:
    try:
        _ = float(cell)
        return True
    except ValueError:
        return False


def cell_length(cell: object) -> int:
    if cell is None:
        return 0
    elif isinstance(cell, tuple):
        return max(cell_length(e) for e in cell)
    elif (isinstance(cell, str) or isinstance(cell, float)) and is_float(cell):
        return len(f"{float(cell):.2f}")
    else:
        return len(str(cell))


def add_space(len: float) -> float:
    return max(10, min(len + 5, len * 1.1))


def dump_excel(
    df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str, index: bool = False
):
    df.to_excel(writer, sheet_name=sheet_name, index=index)
    start_loc = 0
    if index:
        column_length = max(df.index.map(cell_length).max(), cell_length(df.index.name))
        column_length = add_space(column_length)
        writer.sheets[sheet_name].set_column(start_loc, start_loc, column_length)
        start_loc += 1
    for i, column in enumerate(df):
        column_length = max(df[column].map(cell_length).max(), cell_length(column))
        column_length = add_space(column_length)
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
                  score: first(.run_log[] | select(.msg="Final solution") | .args.score),
                }
                """
            )
            .input_text(input.read())
            .all()
        )
        df = pd.DataFrame.from_dict(results)
        dump_excel(df, out, "raw")
        df = df.pivot_table(
            ["mofi", "total_fs", "score"],
            "instance",
            "router",
            sort=False,
            margins_name="Avg.",
            margins=True,
        )
        dump_excel(df, out, "pivot", index=True)


if __name__ == "__main__":
    pivot_results(sys.stdin, sys.stdout.buffer)
