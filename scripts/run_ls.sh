#!/bin/sh

uv run -m disaster_routing.main instance.path=$1 +solver=ls +solver.base=two_phase +solver.base.dsa_solver=${2:-npm} +solver.base.router=${3:-greedy} hydra.verbose=true
