"""Where two recorded agents differ: by cost item, by grid, by fab and by part of the episode, on any network.

    uv run python lab/anastasiia/regime_lab/where.py hub f_none f_all --task=full --entropy=444 --episodes=8

Reads ``play.py``'s records and prints every tag against the first one, in bn USD an episode and as a share of the
room between the naive rule and the clairvoyant plan (the unit of RSS). Shed load is priced at the grid's value of
lost load. A fab's lots are priced at the penalty of the chip they become, which is what a lot is worth only while
that chip is short: an upper estimate, the exact item is ``shortage``. ``offline`` as a tag reads the full-knowledge
plan of ``plan.py starts`` (``outputs/regime_lab/descend``). The numbers are sums of recorded costs, with no sampling
in them: an item of one sign in every episode is a mechanism, and the interval of a score says nothing about it.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import core  # noqa: E402
import plan  # noqa: E402


ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
SHOWN = ("shed", "shortage", "freight", "tariff", "holding", "disposal")  # queue_holding is a part of holding


def _offline(task: str, entropy: int, n: int, start: str = "hybrid") -> dict | None:
    path = plan.OUT / "descend" / f"{task}_{entropy}_{n}_{start}.pkl"
    if not path.is_file():
        return None
    recs, J = core.Episode.of(task, entropy, n).simulate(pickle.loads(path.read_bytes())["acts"])
    return {
        "n": n, "J": int(J), "cpu": [0.0],
        "costs": np.array([[getattr(r.costs, c) for c in ITEMS] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
    }


def _records(tag: str, task: str, entropy: int, ns: list) -> dict:
    if tag == "offline":
        return {n: r for n in ns if (r := _offline(task, entropy, n)) is not None}
    return pickle.loads((plan.OUT / "play" / f"{tag}_{task}_{entropy}.pkl").read_bytes())


def main(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 16, first: int = 0, parts: int = 4, sites: bool = True) -> None:
    ns = list(range(first, first + episodes))
    refs = plan.references(task, entropy, first + episodes)
    data = {tag: _records(tag, task, entropy, ns) for tag in tags}
    ns = [n for n in ns if all(n in d for d in data.values()) and refs[n]["J_oracle_cents"] is not None]
    room = sum(refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns) / 100.0  # USD
    inst, _marks = core.world(task, entropy, ns[0])
    name = [nd.id for nd in inst.nodes]
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    price = {d.k: d.pi for d in inst.demands}
    final = {inst.commodity_index[f"{c}_raw"]: inst.commodity_index[c] for c in ("chip_le", "chip_mat")}
    worth = np.array([price[final[inst.nodes[f].fab.product]] for f in inst.fabs])
    T, E = inst.T, len(ns)
    cut = np.linspace(0, T, parts + 1).astype(int)
    base = data[tags[0]]

    def score(tag: str) -> float:
        return 1.0 - (sum(data[tag][n]["J"] - refs[n]["J_oracle_cents"] for n in ns) / 100.0) / room

    print(f"{task}, root {entropy}, {E} episodes ({ns[0]}..{ns[-1]}); room {room / E / 1e9:.0f} bn USD an episode; + is money saved against {tags[0]}")
    print(f"{tags[0]}: score {score(tags[0]):.4f}, {(1 - score(tags[0])) * room / E / 1e9:.0f} bn an episode above the clairvoyant plan")
    for tag in tags[1:]:
        d = data[tag]
        saved = np.array([base[n]["costs"] - d[n]["costs"] for n in ns])  # episodes x weeks x items
        tot = sum(base[n]["J"] - d[n]["J"] for n in ns) / 100.0
        print(f"\n{tag}: score {score(tag):.4f}, {tot / room:+.4f} against {tags[0]} ({tot / E / 1e9:+.1f} bn an episode)")
        print("  item        bn/episode   of room   episodes with a saving")
        for c in SHOWN:
            s = saved[:, :, ITEMS.index(c)].sum(1)
            print(f"  {c:10s} {s.sum() / E / 1e9:+10.1f}   {s.sum() / room:+.4f}   {int((s > 0).sum())} of {E}")
        print("  part of the episode (weeks): " + "  ".join(
            f"{cut[i] + 1}-{cut[i + 1]}: shed {saved[:, cut[i]:cut[i + 1], 7].sum() / room:+.4f} shortage {saved[:, cut[i]:cut[i + 1], 5].sum() / room:+.4f}"
            for i in range(parts)))
        if not sites:
            continue
        shed = np.array([(base[n]["shed"] - d[n]["shed"]).sum(0) for n in ns]) * voll  # episodes x grids, USD
        lots = np.array([(d[n]["lots"] - base[n]["lots"]).sum(0) for n in ns])  # episodes x fabs, lots more than the base
        print("  grid          shed saved, of room   fabs: lots more than the base, as a share of capacity, and their worth of room")
        for gi, g in enumerate(inst.grids):
            line = f"  {name[g]:10s} {shed[:, gi].sum() / room:+.4f} ({int((shed[:, gi] > 0).sum())}/{E})   "
            for fi, f in enumerate(inst.fabs):
                if inst.nodes[f].fab.grid == g:
                    line += f"{name[f][4:]} {lots[:, fi].sum() / (E * T * inst.nodes[f].fab.cap0):+.3f} {lots[:, fi].sum() * worth[fi] / room:+.4f}  "
            print(line)


if __name__ == "__main__":
    fire.Fire(main)
