#!/bin/sh
instances=$(find instances)
pattern=""
extra_args=""

for arg in "$@"; do
    case $arg in
        --pattern=*)
            pattern="${arg#*=}"
            ;;
        *)
            extra_args="$extra_args $arg"
            ;;
    esac
done

if [ -n "$pattern" ]; then
    instances=$(echo "$instances" | grep -E "$pattern")
fi

uv run -m disaster_routing.main -m instance.path=$(echo "$instances" | paste -sd,) $extra_args
