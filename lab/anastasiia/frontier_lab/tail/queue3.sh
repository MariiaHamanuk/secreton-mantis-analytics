#!/bin/zsh
# The tail lab's third round of 10 October, after ``end_sight`` went into hazard_lab/src: the identity gate again, the
# lab's source against this lab's patch with the switch on, the switch on Small, and the end credit beside it on Full.
#     zsh lab/anastasiia/frontier_lab/tail/queue3.sh <step> ...     (gate3, small3, both3; in the order given)
set -u
cd "$(dirname "$0")/../../../.."
A=outputs/hazard_lab/agents
L=outputs/tail_lab/logs
PLAY="lab/anastasiia/hazard_lab/play.py"
BUILD="lab/anastasiia/hazard_lab/build.py"
T=lab/anastasiia/frontier_lab/tail
WATCH='{"weather_closure": 0.5, "port_strike_stoppage": 0.5, "port_strike_slowdown": 0.5}'
ROOM='{"weeks": 1000000000.0, "floor": 0.0}'
mkdir -p $L

build() {  # name, preset, other numbers: hazard3's settings on the lab's source
    name=$1; preset=$2; shift 2
    [ -d $A/$name ] || uv run python $BUILD $name --preset=$preset --watch="$WATCH" --watch_ask=0.3 --ask_scale=3 \
        --terms='{"model": {}}' "$@" >> $L/build.log 2>&1
}

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
gate3)  # the switches off: the record's actions; ``end_sight`` on: three whole episodes as this lab's patch plays them
    build tw_srcf2 full
    build tw_sx20 full --horizon=20 --hull_until=6 --end_sight=26
    nice -n 10 uv run python lab/anastasiia/hazard_lab/same.py check $A/tw_srcf2 --task=full --episode=3 --weeks=30 > $L/same_full2.log 2>&1 &
    nice -n 10 uv run python lab/anastasiia/hazard_lab/same.py check $A/tw_srcf2 --task=small --episode=9 --weeks=52 > $L/same_small2.log 2>&1 &
    wait
    play tw_sx20 tw_sx20 full 444 3
    {
        cat $L/same_full2.log $L/same_small2.log
        uv run python -c "
import pickle
a = pickle.load(open('outputs/hazard_lab/play/tw_x20_full_444.pkl', 'rb'))
b = pickle.load(open('outputs/hazard_lab/play/tw_sx20_full_444.pkl', 'rb'))
for n in sorted(b):
    print('Full 444 episode', n, 'the patch of this lab', a[n]['J'], 'the lab source with end_sight', b[n]['J'], 'the same' if a[n]['J'] == b[n]['J'] else 'DIFFERS')
"
    } > $L/gate3_result.log 2>&1
    ;;
small3)  # Small 444, episodes 0-63: the window of 20 weeks that plans to the episode's end once it is 26 weeks away
    build tw_x20s small --horizon=20 --hull_until=6 --end_sight=26
    play tw_x20s tw_x20s small 444 64
    {
        show small 444 64 h3_s tw_w20s tw_e20s tw_x20s
        show small 444 64 tw_w20s tw_x20s
    } > $L/small3_result.log 2>&1
    ;;
both3)  # the end credit beside ``end_sight``: Full 444, episodes 0-15, where the credit alone had its largest effect
    build tw_y20 full --horizon=20 --hull_until=6 --end_sight=26 --end_reach --chip_room="$ROOM"
    play tw_y20 tw_y20 full 444 16
    {
        show full 444 16 h3_f tw_w20 tw_e20 tw_x20 tw_y20
        show full 444 16 tw_x20 tw_y20
    } > $L/both_result.log 2>&1
    ;;
esac
echo "$step done $(date)" >> $L/queue.log
done
