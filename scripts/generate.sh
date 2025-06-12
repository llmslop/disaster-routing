#!/bin/sh

for i in $(seq 1 $1);
do
	python -m disaster_routing.main -m instance_gen.output_path="instances/nsfnet-10-$i.json"
done
