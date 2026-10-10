#!/bin/zsh
# The tail lab's plays of 10 October after the port into hazard_lab/src: three processes at a time, nice 10, and each
# step's tables written beside its logs once its plays are over (no table is made while a play runs).
#     zsh lab/anastasiia/frontier_lab/tail/queue.sh <step> ...     (gate, parts, root, base, small; in the order given)
# Logs and tables: outputs/tail_lab/logs/. A play that is kept is not played again (hazard_lab/play.py).
set -u
cd "$(dirname "$0")/../../../.."
A=outputs/hazard_lab/agents
L=outputs/tail_lab/logs
PLAY="lab/anastasiia/hazard_lab/play.py"
PARTS="lab/anastasiia/frontier_lab/tail/parts.py"
BUILD="lab/anastasiia/hazard_lab/build.py"
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
    uv run python $PARTS "$@" --task=$task --entropy=$root --episodes=$n 2>&1 | grep -v Warning
}

for step in "$@"; do
case $step in
gate)  # the switches off: the lab's source acts as the record does
    build tw_srcf full
    nice -n 10 uv run python lab/anastasiia/hazard_lab/same.py check $A/tw_srcf --task=full --episode=3 --weeks=30 > $L/same_full.log 2>&1 &
    nice -n 10 uv run python lab/anastasiia/hazard_lab/same.py check $A/tw_srcf --task=small --episode=9 --weeks=52 > $L/same_small.log 2>&1 &
    wait
    cat $L/same_full.log $L/same_small.log > $L/gate_result.log
    ;;
equiv)  # the switches on: the lab's source plays three whole episodes as this lab's patch on h3_f's files does
    build tw_s20 full --horizon=20 --hull_until=6 --end_reach --chip_room="$ROOM"
    play tw_s20 tw_s20 full 444 3
    uv run python -c "
import pickle
a = pickle.load(open('outputs/hazard_lab/play/tw_e20_full_444.pkl', 'rb'))
b = pickle.load(open('outputs/hazard_lab/play/tw_s20_full_444.pkl', 'rb'))
for n in sorted(b):
    print('Full 444 episode', n, 'the patch of this lab', a[n]['J'], 'the lab source with the same switches', b[n]['J'], 'the same' if a[n]['J'] == b[n]['J'] else 'DIFFERS')
" > $L/equiv_result.log 2>&1
    ;;
parts)  # which switch carries it: Full 444, episodes 0-15
    play tw_r20 tw_r20 full 444 16
    play tw_m20 tw_m20 full 444 16
    {
        show full 444 16 h3_f tw_w20 tw_r20 tw_m20 tw_e20
        show full 444 16 tw_w20 tw_r20 tw_m20 tw_e20
        show full 444 16 tw_e20 tw_r20 tw_m20
        show full 444 48 h3_f tw_e20
        show full 444 48 tw_w20 tw_e20
    } > $L/parts_result.log 2>&1
    ;;
root)  # the root or the clock: the pair without the clock on Full 111, episodes 0-31, beside the pair under the meter
    play h3w20_f tw_w20 full 111 32
    play tw_e20 tw_e20 full 111 32
    {
        show full 111 32 tw_w20 tw_e20
        echo "### under the meter, the same episodes"
        uv run python $PLAY show twc0 twc1 --task=full --entropy=111 --episodes=32 2>&1 | grep -v Warning
        uv run python $PLAY show twc0 twc1 --task=full --entropy=111 --episodes=64 2>&1 | grep -v Warning
    } > $L/root_result.log 2>&1
    ;;
base)  # the window of 26 weeks on the same episodes
    play h3_f h3_f full 111 32
    show full 111 32 h3_f tw_w20 tw_e20 > $L/base_result.log 2>&1
    ;;
small)  # Small 444, episodes 0-63, Small's settings: the credit in the window of 26 weeks, in one of 20, and that one bare
    build tw_e26s small --end_reach --chip_room="$ROOM"
    build tw_e20s small --horizon=20 --hull_until=6 --end_reach --chip_room="$ROOM"
    build tw_w20s small --horizon=20 --hull_until=6
    play tw_e26s tw_e26s small 444 64
    play tw_e20s tw_e20s small 444 64
    play tw_w20s tw_w20s small 444 64
    {
        show small 444 64 h3_s tw_e26s tw_w20s tw_e20s
        show small 444 64 tw_w20s tw_e20s
    } > $L/small_result.log 2>&1
    ;;
esac
echo "$step done $(date)" >> $L/queue.log
done
