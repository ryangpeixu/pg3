#!/bin/bash

EXEC=python #python

START_SEED=0
NUM_SEEDS=1

ENV="spannerlearning" #hiking or trapnewspapers or manymiconic or manygripper or manyferry or spannerlearning
APPROACH="look_ahead" #look_ahead
    
for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # Integrated canonical match
    $EXEC main.py --env $ENV --approach $APPROACH --seed $SEED --train_score_function_name integrated_canonical_match

    # Direct policy eval
    $EXEC main.py --env $ENV --approach $APPROACH --seed $SEED --train_score_function_name direct_policy_eval

    # Primitive canonical match
    $EXEC main.py --env $ENV --approach $APPROACH --seed $SEED --train_score_function_name primitive_canonical_match

done
