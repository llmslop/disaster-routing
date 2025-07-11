#!/bin/sh
instances=$(find instances)
if [ -n "$1" ]; then
    instances=$(grep -E $1 <<< "$instances")
fi

uv run -m disaster_routing.main -m instance.path=$(paste -sd, <<< "$instances") +router=greedy,flow,greedy+ls,flow+ls
