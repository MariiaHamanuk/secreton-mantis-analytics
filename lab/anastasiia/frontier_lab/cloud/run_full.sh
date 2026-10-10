#!/bin/bash
cd ~/repo; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH=$HOME/.local/bin:$PATH
run() { uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/$1 --tag=$2 --task=full --entropy=444 --episodes=16 --n_jobs=16 > outputs/logs/$2.log 2>&1; }
run h3_f h3c_f & run truthall_f truthallc_f & wait
run h3hz_f h3hzc_f & run tah0_f tah0c_f & wait
echo done > outputs/logs/ALL_DONE
