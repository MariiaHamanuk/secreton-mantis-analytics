"""A repair's kept play beside the base's: the paired score, where the money moved, and what the machine did.

    uv run python lab/anastasiia/frontier_lab/diag/repair.py h3_f dg_hold_f --first=16 --episodes=32
    uv run python lab/anastasiia/frontier_lab/diag/repair.py h3_s dg_hold_s --task=small --episodes=64
    uv run python lab/anastasiia/frontier_lab/diag/repair.py h3_f dg_hold_f --episodes=48 --other=h3c_f --noby_episode

Reads the plays of ``hazard_lab/play.py``. Beside the paired difference (the board's weighting with the episodes
drawn again inside each harm level, and all saved over all room) it prints, per episode, the first week in which
the two plays send anything different and whether that difference is on the valves alone. The repairs of this folder
act on the valves from terminals into grids and nowhere else, so a first difference elsewhere is not the repair: it is
the solver's wall-clock limits under another load of the machine, and from that week the pair compares two machines
as well as two agents. The CPU seconds of the weeks before the first difference, when both agents do the same work
from the same state, say how much of a difference in the week's time is the machine's.
"""

import fire
import numpy as np
from common import BN, ITEMS, WEIGHTS, Names, kept, paired, refs


def main(base: str, new: str, task: str = "full", entropy: int = 444, first: int = 0, episodes: int = 16,
         by_episode: bool = True, other: str = "") -> None:  # fmt: skip
    nm = Names(task)
    inst, N = nm.inst, nm.inst.nodes
    A, B = kept(base, task, entropy), kept(new, task, entropy)
    R = refs(task, entropy, first + episodes)
    E = [n for n in range(first, first + episodes) if n in A and n in B]
    if not E:
        print("no episode is kept for both")
        return
    lvl = np.array([R[n]["stratum"] for n in E])
    room = np.array([R[n]["J_naive_cents"] - R[n]["J_oracle_cents"] for n in E]) / 1e11
    naive = np.array([R[n]["J_naive_cents"] for n in E]) / 1e11
    Ja, Jb = np.array([A[n]["J"] for n in E]) / 1e11, np.array([B[n]["J"] for n in E]) / 1e11
    saved = Ja - Jb
    valves = np.array([
        lane is None and N[inst.edges[e].head].grid is not None and N[inst.edges[e].tail].id.startswith("term")
        for e, _k, lane in inst.action_slots
    ])  # fmt: skip
    T = A[E[0]]["sent"].shape[0]

    rng = np.random.default_rng(0)
    pooled = [saved[i].sum() / room[i].sum() for i in (rng.integers(0, len(E), len(E)) for _ in range(4000))]
    print(
        f"{new} against {base}: {task}, root {entropy}, {len(E)} episodes ({E[0]}..{E[-1]}), by harm level "
        f"{[int((lvl == s).sum()) for s in (1, 2, 3, 4)]}"
    )
    if set(lvl.tolist()) == set(WEIGHTS):
        point, lo, hi = paired(saved, lvl, room)
        whole = sum(WEIGHTS[s] * room[lvl == s].mean() for s in WEIGHTS)
        score = [sum(WEIGHTS[s] * (naive - J)[lvl == s].mean() for s in WEIGHTS) / whole for J in (Ja, Jb)]
        print(f"  the board's weighting: {score[0]:.4f} > {score[1]:.4f}, {point:+.4f} ({lo:+.4f} to {hi:+.4f})")
    print(
        f"  all saved over all room: {saved.sum() / room.sum():+.4f} ({np.percentile(pooled, 5):+.4f} to {np.percentile(pooled, 95):+.4f}); "
        f"{saved.mean():+.1f} bn an episode, cheaper in {int((saved > 0.05).sum())} of {len(E)}, the same to 0.05 bn in "
        f"{int((np.abs(saved) <= 0.05).sum())}, worst {saved.min():+.1f}, best {saved.max():+.1f}"
    )
    for s in sorted(set(lvl.tolist())):
        m = lvl == s
        print(
            f"    level {s} ({m.sum()} episodes): {saved[m].mean():+.1f} bn an episode, {saved[m].sum() / room[m].sum():+.4f} of its room"
        )

    items = np.array([(A[n]["costs"] - B[n]["costs"]).sum(axis=0) for n in E]) / BN  # saved, by item
    print(
        "  saved by item, bn an episode: "
        + ", ".join(f"{ITEMS[i]} {items[:, i].mean():+.2f}" for i in (5, 7, 6, 3))
        + f", the other four {np.delete(items, (5, 7, 6, 3), axis=1).sum(axis=1).mean():+.2f}"
    )
    voll = np.array([N[g].grid.voll for g in inst.grids])
    shed = np.array([((A[n]["shed"] - B[n]["shed"]) * voll).sum(axis=0) for n in E]) / BN
    print("  shed saved by grid: " + ", ".join(f"{x[5:]} {v:+.1f}" for x, v in zip(nm.grids, shed.mean(axis=0))))
    blocks = [(i, min(i + 13, T)) for i in range(0, T, 13)]
    weekly = np.array([(A[n]["costs"] - B[n]["costs"]).sum(axis=1) for n in E]) / BN
    print("  saved by 13-week block: " + " ".join(f"{weekly[:, lo:hi].sum(axis=1).mean():+.1f}" for lo, hi in blocks))

    firsts, clean, cpu_before = [], [], []
    rows = []
    for i, n in enumerate(E):
        diff = np.abs(A[n]["sent"] - B[n]["sent"]) > 1e-6 * np.maximum(1.0, np.abs(A[n]["sent"]))
        weeks = np.flatnonzero(diff.any(axis=1))
        t0 = int(weeks[0]) if len(weeks) else T
        on_valves = bool(len(weeks)) and not diff[t0, ~valves].any()
        firsts.append(t0)
        clean.append(on_valves or t0 == T)
        ca, cb = np.array(A[n]["cpu"]), np.array(B[n]["cpu"])
        if t0 >= 4:
            cpu_before.append((ca[1:t0].sum(), cb[1:t0].sum()))  # week 1 has the agent's start-up
        notes = B[n]["notes"] or {}
        rows.append(
            f"    ep {n:3d} L{lvl[i]}: saved {saved[i]:+7.1f} (unmet demand {items[i, 5]:+6.1f}, shed {items[i, 7]:+6.1f}); "
            f"first week sent differs {t0 + 1 if t0 < T else '-'}{'' if t0 == T else ' on valves only' if on_valves else ' NOT on valves'}; "
            f"moved in {notes.get('topped_qty', 0.0) / 1e3:.0f}k, held back {notes.get('cut_qty', 0.0) / 1e3:.0f}k; "
            f"cpu median {np.median(cb):.2f} against {np.median(ca):.2f}"
        )
    clean = np.array(clean)
    print(
        f"  the first difference in what is sent is on the valves alone (or there is none) in {int(clean.sum())} of {len(E)} episodes; "
        f"saved there {saved[clean].mean() if clean.any() else float('nan'):+.1f} bn an episode, in the other {int((~clean).sum())}: "
        f"{saved[~clean].mean() if (~clean).any() else float('nan'):+.1f}"
    )
    ca, cb = np.concatenate([A[n]["cpu"] for n in E]), np.concatenate([B[n]["cpu"] for n in E])
    print(
        f"  CPU a week: median {np.median(cb):.2f} against {np.median(ca):.2f} s, 95th percentile {np.percentile(cb, 95):.2f} against "
        f"{np.percentile(ca, 95):.2f}, longest {cb.max():.2f} against {ca.max():.2f}"
    )
    if cpu_before:
        before = np.array(cpu_before)
        print(
            f"  CPU of the weeks before the first difference (the same work from the same state; {len(before)} episodes, "
            f"{int(sum(max(0, t - 1) for t in firsts if t >= 4))} weeks): {before[:, 1].sum() / before[:, 0].sum():.3f} of the base's; "
            f"of all weeks: {cb.sum() / ca.sum():.3f}"
        )
    wa = np.array([[np.mean(A[n]["cpu"][lo:hi]) for lo, hi in blocks] for n in E]).mean(axis=0)
    wb = np.array([[np.mean(B[n]["cpu"][lo:hi]) for lo, hi in blocks] for n in E]).mean(axis=0)
    print("  CPU a week by 13-week block: " + " ".join(f"{b:.2f}/{a:.2f}" for a, b in zip(wa, wb)))
    if other:  # a second kept play of the base's own settings: what one play of the base is worth as a yardstick
        C = kept(other, task, entropy)
        if all(n in C for n in E) and set(lvl.tolist()) == set(WEIGHTS):
            Jc = np.array([C[n]["J"] for n in E]) / 1e11
            both = 0.5 * (Ja + Jc)
            print(
                f"  against {other}: {'%+.4f (%+.4f to %+.4f)' % paired(Jc - Jb, lvl, room)}; against the mean of {base} and {other}: "
                f"{'%+.4f (%+.4f to %+.4f)' % paired(both - Jb, lvl, room)}"
            )
            print(
                f"  {other} against {base}: {'%+.4f (%+.4f to %+.4f)' % paired(Ja - Jc, lvl, room)}; they differ by "
                f"{np.abs(Ja - Jc).mean():.1f} bn an episode in absolute mean, {np.abs(Ja - Jc).max():.1f} at most; correlation over the "
                f"episodes of what {new} saves with what {other} saves: {np.corrcoef(saved, Ja - Jc)[0, 1]:+.2f}"
            )
    if by_episode:
        print("\n".join(rows))


if __name__ == "__main__":
    fire.Fire(main)
