#!/bin/sh
if [ -n "$1" ]; then
    filename=$(mktemp --suffix ".$1")
else
    filename=$(mktemp)
fi
cat > "$filename" && scripts/open.sh "$filename"
