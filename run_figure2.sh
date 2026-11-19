#!/bin/bash

# Reproduce Figure 2 (overestimation bias) on a MuJoCo task.
# Trains DDPG, DDQN-AC, DQ-AC and CDQ/TD3, then builds the figure.
# Defaults to Hopper-v5, 200k steps, seed 0. Override: ENV=Walker2d-v5 ./run_figure2.sh

ENV="${ENV:-Hopper-v5}"
STEPS="${STEPS:-200000}"
SEED="${SEED:-0}"

for POLICY in DDPG DDQN_AC DQ_AC TD3
do
	python overestimation.py \
	--policy "$POLICY" \
	--env "$ENV" \
	--seed "$SEED" \
	--max_timesteps "$STEPS"
done

python plot_figure2.py --env "$ENV" --no_show
