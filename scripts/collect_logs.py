#!/bin/env python3

import json
import sys
from collections.abc import Iterable
from os import sep

from extract_log import collect_log
from fs import open_fs as fs_open_fs
from fs.base import FS
from fs.errors import ResourceNotFound
from yaml import safe_load


def collect_logs(fses: Iterable[FS]):
    sys.stdout.write("[")
    first = True
    for fs in fses:
        for date in fs.listdir("."):
            if date == "results":
                continue
            for time in fs.listdir(date):
                group = f"{date}T{time}"
                for run in fs.listdir(date + sep + time):
                    dir = date + sep + time + sep + run
                    if fs.isdir(dir):
                        try:
                            config = safe_load(fs.open(f"{dir}/.hydra/config.yaml"))
                            run_log = collect_log(fs.open(f"{dir}/main.log"))
                            if not first:
                                sys.stdout.write(",")
                            first = False
                            json.dump(
                                {
                                    "group": group,
                                    "config": config,
                                    "run_log": run_log,
                                },
                                sys.stdout,
                            )
                        except ResourceNotFound:
                            continue
    sys.stdout.write("]")


def open_fs(path: str) -> FS:
    if path.startswith("gh://"):
        return ...
    else:
        return fs_open_fs(path)


if __name__ == "__main__":
    collect_logs([open_fs(path) for path in sys.argv[1:]])
