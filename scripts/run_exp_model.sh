#!/bin/bash

cd ..

method=dagger
teacher=twodifdetm

declare -a n_samples=("500")

for n in "${n_samples[@]}"; do
    command="python -u active_main.py -query_pol $method -teacher $teacher -n_samples $n -n_train 200"
    echo $command 
    $command > log/submission/${method}_model_${teacher}_${n}.txt
done

