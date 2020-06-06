#!/bin/bash

cd ..

method=$1
threshold=${2:-0}

declare -a teachers=("detm" "rand" "tworand" "twodifdetm")

for teacher in "${teachers[@]}"; do
    command="python -u active_main.py -query_pol $method -teacher $teacher -threshold $threshold"
    echo $command 
    $command > log/submission/${method}_model_${teacher}.txt
done

