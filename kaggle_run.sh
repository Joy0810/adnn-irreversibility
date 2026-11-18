#!/usr/bin/env bash
# Runs every GPU experiment for one model on Kaggle (2x T4). See README, "Kaggle".
#   bash kaggle_run.sh qwen7b main      # E1, E2 (CPU, in parallel), E4, E5, E6, E7   ~6-7 h
#   bash kaggle_run.sh qwen7b stated    # E3                                          ~3-4 h
# Every step resumes if the session restarts and the same command is run again.
set -x
M=$1; PART=${2:-main}
ROWS="data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl"
mkdir -p logs

if [ "$PART" = stated ]; then
  python -m experiments.stated --model $M --rows $ROWS 2>&1 | tee logs/${M}_E3.log
  exit
fi

python -m models.extract --model $M --rows $ROWS 2>&1 | tee logs/${M}_E1.log
python -m probes.train --model $M --rows $ROWS --jobs 3 > logs/${M}_E2.log 2>&1 &   # CPU while the GPU runs E4/E5
E2=$!
python -m experiments.agent_runs --model $M 2>&1 | tee logs/${M}_E4.log
python -m experiments.agent_runs --model $M --careful 2>&1 | tee logs/${M}_E5.log
wait $E2; tail -3 logs/${M}_E2.log
python -m experiments.steering --model $M --mode decision --alphas -2 -1 0 1 2 2>&1 | tee logs/${M}_E6.log
python -m experiments.steering --model $M --mode rollouts --alphas -2 0 2 --limit 20 2>&1 | tee logs/${M}_E7.log
