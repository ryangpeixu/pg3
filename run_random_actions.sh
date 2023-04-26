#!/bin/bash

EXEC=python #python

START_SEED=0
NUM_SEEDS=10

ENV="heavypack" #hiking or trapnewspapers or manymiconic or manygripper or manyferry or spannerlearning
APPROACH="random_actions" #look_ahead
    
for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    $EXEC main.py --env $ENV --approach $APPROACH --seed $SEED  --train_score_function_name integrated_canonical_match

done
