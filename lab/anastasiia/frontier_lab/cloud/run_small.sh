#!/bin/bash
# $1, $2: the first and the last episode of root 444 this machine plays
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
F=$1; L=$2; N=$((L-F+1))
uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/h3_s --tag=h3c_s --task=small --entropy=444 --first=$F --episodes=$N --n_jobs=11 > outputs/logs/h3c_s.log 2>&1 &
uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/h3p3_s --tag=h3p3c_s --task=small --entropy=444 --first=$F --episodes=$N --n_jobs=11 > outputs/logs/h3p3c_s.log 2>&1 &
for i in $(seq 1 32); do
  uv run python lab/anastasiia/frontier_lab/scen/play.py run two8b two8bp two4b two2b two8be two8bn two16b --episodes=$F-$L > outputs/logs/w$i.log 2>&1 &
  sleep 0.5
done
wait
echo done > outputs/logs/ALL_DONE
