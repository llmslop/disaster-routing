#!/bin/sh

uv run -m disaster_routing.main -m instance.path=`ls instances/nsfnet-*.json | paste -sd,` +router=greedy,flow,greedy+ls
