#!/bin/env python3
import sys
import json
from fs import open_fs
from os import sep
from fs.errors import ResourceNotFound
from yaml import safe_load

from extract_log import collect_log

fs = open_fs(sys.argv[1])
sys.stdout.write("[")
first = True
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
