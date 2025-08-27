#!/bin/sh

uv run -m disaster_routing.main instance.path=$1 +solver=two_phase +solver.dsa_solver=${2:-npm} +solver.router=${3:-greedy} hydra.verbose=true
