#!/bin/bash
# the final agent against itself with one rule off, 64 episodes of root 111 (paired)
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics
WT=$MAIN/.claude/worktrees/agent-a9fc6562ebf34a489
args=""
for v in abl_pack abl_raw abl_wafer abl_endgame abl_strait abl_queue abl_cover abl_waits abl_useful abl_w1 abl_margin; do
  args="$args $WT/lab_scratch/var/$v"
done
cd $WT
$MAIN/.venv/bin/python $MAIN/outputs/heur_lab/tools/compare.py --base=$WT/lab_scratch/var/final --episodes=64 --n_jobs=3 $args > $WT/lab_scratch/abl64.txt 2>&1
echo finished >> $WT/lab_scratch/abl64.txt
