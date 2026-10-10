#!/bin/bash
# $1 $2: Full episodes: the conditional two-scenario program first, then the model with a window of 20 and of 16 weeks
# and the lean scenario program over the windows of 26, 20 and 16 weeks
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
tar xzf ~/upd3.tgz -C ~/repo 2>/dev/null
F=$1; L=$2; N=$((L-F+1))
for i in $(seq 1 32); do
  uv run python lab/anastasiia/frontier_lab/scen/play.py run two2c two2l two1l two2l20 two2l16 two1l20 two1l16 --task=full --episodes=$F-$L > outputs/logs/j_$i.log 2>&1 &
  sleep 0.3
done
for W in 20 16; do
  uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/h3w${W}_f --tag=h3w${W}c_f --task=full --entropy=444 --first=$F --episodes=$N --n_jobs=8 > outputs/logs/h3w${W}c_f.log 2>&1 &
done
wait
echo done > outputs/logs/ISO_DONE
