#!/bin/bash
# $1: a file to wait for in outputs/logs ("-": none), $2: entropy, $3 $4: Small episodes, $5: a name for the logs,
# the rest: scenario variants. The model itself is played first (tag h3c_s).
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
[ "$1" != "-" ] && while [ ! -f outputs/logs/$1 ]; do sleep 10; done
tar xzf ~/upd4.tgz -C ~/repo 2>/dev/null
ENT=$2; F=$3; L=$4; NAME=$5; N=$((L-F+1)); shift 5
uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/h3_s --tag=h3c_s --task=small --entropy=$ENT --first=$F --episodes=$N --n_jobs=16 > outputs/logs/${NAME}_base.log 2>&1 &
for i in $(seq 1 32); do
  uv run python lab/anastasiia/frontier_lab/scen/play.py run "$@" --task=small --entropy=$ENT --episodes=$F-$L > outputs/logs/${NAME}_$i.log 2>&1 &
  sleep 0.3
done
wait
echo done > outputs/logs/${NAME}_DONE
