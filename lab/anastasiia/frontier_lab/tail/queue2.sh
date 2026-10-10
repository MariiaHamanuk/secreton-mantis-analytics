#!/bin/zsh
# The tail lab's second round of 10 October: what is left of the short window's loss in weeks 79 to 90, and a window
# that is the rest of the episode once its end is in sight (``end_sight``). Three processes at a time, nice 10.
#     zsh lab/anastasiia/frontier_lab/tail/queue2.sh <step> ...     (diag, sight111, sight444; in the order given)
set -u
cd "$(dirname "$0")/../../../.."
A=outputs/hazard_lab/agents
L=outputs/tail_lab/logs
PLAY="lab/anastasiia/hazard_lab/play.py"
T=lab/anastasiia/frontier_lab/tail
mkdir -p $L

play() {  # folder, tag, task, root, episodes
    nice -n 10 uv run python $PLAY run $A/$1 --tag=$2 --task=$3 --entropy=$4 --episodes=$5 --n_jobs=3 >> $L/$2_$3_$4.log 2>&1
}

show() {  # task, root, episodes, base, tags: the paired scores and the loss by part of the episode
    task=$1; root=$2; n=$3; shift 3
    echo "### $* ($task $root, $n episodes)"
    uv run python $PLAY show "$@" --task=$task --entropy=$root --episodes=$n 2>&1 | grep -v Warning
    uv run python $T/parts.py "$@" --task=$task --entropy=$root --episodes=$n 2>&1 | grep -v Warning
}

for step in "$@"; do
case $step in
diag)  # the last weeks by item, and the patch with its new switches off against the record
    {
        uv run python $T/endzone.py h3_f tw_w20 tw_e20 --entropy=111 --episodes=32 2>&1 | grep -v Warning
        uv run python $T/endzone.py h3_f tw_e20 --entropy=444 --episodes=48 2>&1 | grep -v Warning
        uv run python $T/endzone.py h3_f tw_w20 tw_r20 tw_m20 tw_e20 --entropy=444 --episodes=16 2>&1 | grep -v Warning
    } > $L/diag_result.log 2>&1
    uv run python $T/make.py diff >> $L/build.log 2>&1
    [ -d $A/tw_id2 ] || uv run python $T/make.py tw_id2 >> $L/build.log 2>&1
    [ -d $A/tw_x20 ] || uv run python $T/make.py tw_x20 --horizon=20 --hull_until=6 --end_sight=26 >> $L/build.log 2>&1
    nice -n 10 uv run python lab/anastasiia/hazard_lab/same.py check $A/tw_id2 --task=full --episode=3 --weeks=30 > $L/same_id2.log 2>&1
    ;;
sight111)  # the window of 20 weeks that plans to the episode's end from week 79: Full 111, episodes 0-31
    play tw_x20 tw_x20 full 111 32
    {
        show full 111 32 h3_f tw_w20 tw_e20 tw_x20
        show full 111 32 tw_w20 tw_x20
        uv run python $T/endzone.py h3_f tw_x20 --entropy=111 --episodes=32 2>&1 | grep -v Warning
    } > $L/sight111_result.log 2>&1
    ;;
sight444)  # the same on Full 444, episodes 0-15
    play tw_x20 tw_x20 full 444 16
    {
        show full 444 16 h3_f tw_w20 tw_e20 tw_x20
        show full 444 16 tw_w20 tw_x20
        uv run python $T/endzone.py h3_f tw_x20 --entropy=444 --episodes=16 2>&1 | grep -v Warning
    } > $L/sight444_result.log 2>&1
    ;;
esac
echo "$step done $(date)" >> $L/queue.log
done
