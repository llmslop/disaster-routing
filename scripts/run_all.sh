#!/bin/sh
instances=$(find instances)
if [ -n "$1" ]; then
    instances=$(echo "$instances" | grep -E $1)
fi

uv run -m disaster_routing.main -m instance.path=$(echo "$instances" | paste -sd,) +router=greedy,flow,greedy+ls,flow+ls,sga
