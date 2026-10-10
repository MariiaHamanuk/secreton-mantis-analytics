#!/bin/bash
# $1: task, $2: episodes "a-b", $3: workers, the rest: the scenario variants in order
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
T=$1; E=$2; W=$3; shift 3
for i in $(seq 1 $W); do
  uv run python lab/anastasiia/frontier_lab/scen/play.py run "$@" --task=$T --episodes=$E > outputs/logs/t_$1_$i.log 2>&1 &
  sleep 0.5
done
wait
echo done > outputs/logs/TAGS_DONE
