#!/bin/bash
# every distinct variant against pull on 32 episodes of root 111 (one compare.py process: pull is scored once)
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics
WT=$MAIN/.claude/worktrees/agent-a9fc6562ebf34a489
args=""
for v in only_pack only_raw only_wafer all_wcap s1 s2 s3 r1 r2 w1 w2 w3 w4 w5 u1 u2 g1 g2 g3 g4 a1 a2 a3 a4 e1 e3 e4 b1 b2 b3 b4 n1 n2 n3 n4 n5 n6 p3 p4 q1 q2 q3 q4 s_a s_b s_c qe0 k1 m1 m2 m3 m4 x3 x4 t1 f1; do
  args="$args $WT/lab_scratch/var/$v"
done
cd $WT
$MAIN/.venv/bin/python $MAIN/outputs/heur_lab/tools/compare.py --base=$MAIN/agents/pull --episodes=32 --n_jobs=3 $args > $WT/lab_scratch/vs_pull_32.txt 2>&1
echo finished >> $WT/lab_scratch/vs_pull_32.txt
