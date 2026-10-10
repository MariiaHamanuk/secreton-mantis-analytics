#!/bin/bash
# $1: task, $2: episodes, $3: workers, $4: a name for the logs, the rest: variants. The update is unpacked first.
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
tar xzf ~/upd4.tgz -C ~/repo 2>/dev/null
T=$1; E=$2; W=$3; NAME=$4; shift 4
for i in $(seq 1 $W); do
  uv run python lab/anastasiia/frontier_lab/scen/play.py run "$@" --task=$T --episodes=$E > outputs/logs/${NAME}_$i.log 2>&1 &
  sleep 0.3
done
wait
echo done > outputs/logs/${NAME}_DONE
