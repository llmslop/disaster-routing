#!/bin/env python3

import shutil
import os
import sys

dir = "instances/dataset-1"
topologies = {
    "nsfnet": [2, 5, 6, 9, 11],
    # "cost239": [1, 2, 7, 8, 11],
}
instance_sets = [(10, 10), (20, 10), (50, 10), (100, 20)]

# clear directory dir
if "-f" in sys.argv[1:] or "--force" in sys.argv[1:]:
    print(f"Force removing directory {dir}")
    if os.path.exists(dir):
        shutil.rmtree(dir)
os.makedirs(dir, exist_ok=True)

for topology, dc_pos in topologies.items():
    for size, num in instance_sets:
        for inst_num in range(num):
            cmd = (
                "uv run -m disaster_routing.main "
                + f"instance.path={dir}/{topology}-{size:04}-{inst_num:02}.json "
                + f"instance.possible_dc_positions=[{','.join(map(str, dc_pos))}] "
                + f"instance.num_requests={size} "
                + f"instance.topology_name={topology} "
                + "ilp_check=false "
            )
            code = os.system(cmd)
            if code != 0:
                raise RuntimeError(
                    f"Failed to generate instance for {topology} with size {size} and num {num}. Command: {cmd}"
                )
        print(f"Generated {num} instances for topology {topology} with size {size}")
