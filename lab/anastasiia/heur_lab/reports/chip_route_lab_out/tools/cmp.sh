#!/bin/bash
# usage: cmp.sh BASE EPISODES AGENT...   (names under lab_scratch/var, 'pull', or absolute paths)
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics
WT=$MAIN/.claude/worktrees/agent-a9fc6562ebf34a489
base=$1
eps=$2
shift 2
resolve() {
  case "$1" in
    pull) echo $MAIN/agents/pull ;;
    /*) echo $1 ;;
    *) echo $WT/lab_scratch/var/$1 ;;
  esac
}
args=""
for a in "$@"; do args="$args $(resolve $a)"; done
$MAIN/.venv/bin/python $MAIN/outputs/heur_lab/tools/compare.py --base=$(resolve $base) --episodes=$eps --n_jobs=3 $args 2>&1 | tail -n +2 | sed "s#$WT/lab_scratch/var/##; s#$MAIN/agents/##"
