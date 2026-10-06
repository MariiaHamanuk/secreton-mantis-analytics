#!/bin/bash
# the milestone variants against pull on 32 episodes of root 111 (one compare.py process: pull is scored once)
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics
WT=$MAIN/.claude/worktrees/agent-a9fc6562ebf34a489
args=""
for v in r1 w2 w4 u1 a3 e4 n2 q4 s_a qe0 k1 final3; do
  args="$args $WT/lab_scratch/var/$v"
done
cd $WT
$MAIN/.venv/bin/python $MAIN/outputs/heur_lab/tools/compare.py --base=$MAIN/agents/pull --episodes=32 --n_jobs=3 $args > $WT/lab_scratch/vs_pull2.txt 2>&1
echo finished >> $WT/lab_scratch/vs_pull2.txt
