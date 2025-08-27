#!/bin/sh

uv run -m disaster_routing.main instance.path=$1 +solver=sga +solver.dsa_solver=${2:-npm} +solver.approximate_dsa_solver=${3:-fpga} hydra.verbose=true
