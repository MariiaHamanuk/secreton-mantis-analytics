"""Statistics of the LSF search on Full: moves by fuel / type / band, nuclear detail, sign consistency of top slots.

    uv run python lab/anastasiia/mpc_lab/planners/lsf_full_stats.py [--task=full --entropy=444]
Reads outputs/planner_LSF/cache_<task>_<entropy>/lsf_ep*.pkl (accepted-move lists with (week, old, new) changes).
"""

import collections
import glob
import os
import pickle
import re

import fire
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
BN = 1e11


def band(t):
    return min((t - 1) // 26, 3)


BANDS = ["1-26", "27-52", "53-78", "79-104"]


def main(task="full", entropy=444):
    d = os.path.join(ROOT, "outputs", "planner_LSF", f"cache_{task}_{entropy}")
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "lsf_ep*.pkl"))):
        n = int(re.search(r"lsf_ep(\d+)", p).group(1))
        with open(p, "rb") as f:
            res[n] = pickle.load(f)
    ns = sorted(res)
    print("episodes", ns)
    print("\nper episode: J before / after (bn), dJ, accepted, replays, cpu, secs")
    for n in ns:
        r = res[n]
        print(f"{n} {r['before'] / BN:9.2f} {r['after'] / BN:9.2f} {(r['before'] - r['after']) / BN:7.3f} {r['accepted']:5d} "
              f"{r['replays']:6d} {r['cpu']:6.0f} {r['secs']:6.0f} fuel dispatches {r['n_fuel_disp']}")
    # fuel x type x band gain/count
    print("\nmoves / gain bn by fuel, by type, by band (per episode and total)")
    hdr = f"{'ep':>3} " + " ".join(f"{k:>14}" for k in ("lng", "crude", "nucfuel", "valve", "order"))
    print(hdr)
    tot = collections.defaultdict(lambda: [0, 0.0])
    for n in ns:
        c = collections.defaultdict(lambda: [0, 0.0])
        for m in res[n]["moves"]:
            for k in (m["fuel"], m["type"], BANDS[band(m["week"])]):
                c[k][0] += 1
                c[k][1] += m["gain"] / BN
                tot[k][0] += 1
                tot[k][1] += m["gain"] / BN
        print(f"{n:>3} " + " ".join(f"{c[k][0]:>6}/{c[k][1]:6.2f}" for k in ("lng", "crude", "nucfuel", "valve", "order")) + "  | bands "
              + " ".join(f"{c[b][0]}/{c[b][1]:.2f}" for b in BANDS))
    print("TOTAL " + ", ".join(f"{k} {v[0]} moves {v[1]:.2f} bn" for k, v in sorted(tot.items(), key=lambda kv: -kv[1][1])))
    # nuclear detail
    print("\nnuclear moves")
    for n in ns:
        nm = [m for m in res[n]["moves"] if m["fuel"] == "nucfuel"]
        print(f" ep {n}: {len(nm)} moves, gain {sum(m['gain'] for m in nm) / BN:.3f} bn; slots " + str(collections.Counter(m["name"] for m in nm).most_common(4)))
    # by slot: gain per episode, net quantity change per band, sign consistency
    slots = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(g=0.0, k=0, dq=np.zeros(4), early=0.0)))
    for n in ns:
        for m in res[n]["moves"]:
            e = slots[(m["type"], m["fuel"], m["name"])][n]
            e["g"] += m["gain"] / BN
            e["k"] += 1
            for t, a, b in m["changes"]:
                e["dq"][band(t)] += b - a
    rows = []
    for key, per in slots.items():
        g = sum(v["g"] for v in per.values())
        # sign of net quantity change (sum over weeks) per episode; how many episodes agree with the majority
        net = [v["dq"].sum() for v in per.values()]
        pos = sum(x > 0 for x in net)
        neg = sum(x < 0 for x in net)
        rows.append((g, key, per, pos, neg))
    rows.sort(key=lambda r: -r[0])
    print("\ntop slots: total gain bn | episodes with moves | episodes net dQ>0 / <0 | gain per episode | net dQ per band summed over episodes")
    for g, key, per, pos, neg in rows[:25]:
        dq = sum(v["dq"] for v in per.values())
        print(f"{key[0]:5s} {key[1]:7s} {key[2][:62]:62s} {g:7.2f} | {len(per)}/{len(ns)} | +{pos}/-{neg} | "
              + " ".join(f"{per[n]['g']:.1f}" if n in per else "-" for n in ns) + " | " + " ".join(f"{x:,.0f}" for x in dq))
    # grid-level / lane-level aggregation for valves: by grid
    print("\nvalves by grid (gain bn per episode)")
    g2 = collections.defaultdict(lambda: collections.defaultdict(float))
    for n in ns:
        for m in res[n]["moves"]:
            if m["type"] == "valve":
                g2[(m["fuel"], m["name"].split(".")[-1])][n] += m["gain"] / BN
    for k, per in sorted(g2.items(), key=lambda kv: -sum(kv[1].values()))[:15]:
        print(f"  {k[0]:7s} {k[1]:10s} total {sum(per.values()):6.2f} | " + " ".join(f"{per.get(n, 0):.1f}" for n in ns))
    print("\norders by source->fuel route prefix (gain bn per episode)")
    g3 = collections.defaultdict(lambda: collections.defaultdict(float))
    for n in ns:
        for m in res[n]["moves"]:
            if m["type"] == "order":
                g3[(m["fuel"], re.sub(r"\s*\[.*", "", m["name"]))][n] += m["gain"] / BN
    for k, per in sorted(g3.items(), key=lambda kv: -sum(kv[1].values()))[:15]:
        print(f"  {k[0]:7s} {k[1]:52s} total {sum(per.values()):6.2f} | " + " ".join(f"{per.get(n, 0):.1f}" for n in ns))
    # moves
    mv = collections.defaultdict(lambda: [0, 0.0])
    for n in ns:
        for m in res[n]["moves"]:
            mv[m["move"]][0] += 1
            mv[m["move"]][1] += m["gain"] / BN
    print("\nby move: " + ", ".join(f"{k} {v[0]}/{v[1]:.1f}" for k, v in sorted(mv.items(), key=lambda kv: -kv[1][1])))
    # cost components
    keys = list(res[ns[0]]["comp0"])
    print("cost components after-before (bn) per episode:")
    for n in ns:
        print(f" ep {n}: " + ", ".join(f"{k} {(res[n]['comp1'][k] - res[n]['comp0'][k]) / 1e9:+.2f}" for k in keys))
    # direction of changes: total increase vs decrease of fuel quantity per fuel
    print("\nnet quantity change by fuel (sum new-old over all accepted moves; units of commodity):")
    for f in ("lng", "crude", "nucfuel"):
        print(f"  {f}: " + " ".join(f"{sum(b - a for m in res[n]['moves'] if m['fuel'] == f for _, a, b in m['changes']):,.0f}" for n in ns))


if __name__ == "__main__":
    fire.Fire(main)
