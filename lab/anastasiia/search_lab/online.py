"""Step 2: the same search without the future. Each week the network is frozen as the agent sees it.

    uv run python lab/anastasiia/search_lab/online.py agents/anastasiia_hybrid_chiplp --task=small --episodes=16 \
        --moves=fuel --budget=600 --horizon=6 --n_jobs=8

Week by week, in the true episode:

1. the model of the rest of the episode is the network of this week kept for every later week (edge capacities, strait
   openness and throughput, supply, grid output, fab capacity as the observation gives them at the start of the week;
   this week's prohibitions and tariffs; demand at its mean so far; no fab outage that has not happened yet);
2. the plan for weeks t..T (at first the actions the agent played) is searched in that model, the dispatches of weeks
   t .. t + horizon - 1 only, this week's first, ``budget`` replays;
3. week t of the plan is played in the true episode, and the plan goes on to the next week.

What it measures: how much of the gain of ``search.py`` (which knows every week's true network) is left when the
search may only look at "it stays as it is". No limit on time here: the week's CPU budget is step 3.

One thing is not clean: the plan starts from the actions the agent played in the true episode, so the entries of later
weeks already react to events as they happened. The search itself never sees them; a later prohibition simply drops
the entry when its week comes, as the environment would.
"""

import dataclasses
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import SETS, Replay, played, pooled, search, slot_kinds, world  # noqa: E402


NOW = {"u": "u_now", "o": "o_now", "kappa": "kappa_now", "supply": "supply_now", "G_bar": "G_bar_now",
       "y_bar": "y_bar_now", "R": "R_now", "alpha_bar": "alpha_now"}  # fmt: skip
THIS_WEEK = ("c", "tariff", "prohibited", "wr_class", "h_queue", "c_wr", "R_osat", "sigma_scr")
GROUPS = {  # what may be frozen separately (``--freeze``), to see which part of "it stays as it is" misleads
    "power": ("G_bar", "y_bar"),
    "edges": ("u", "c"),
    "straits": ("o", "kappa", "wr_class", "h_queue"),
    "bans": ("prohibited", "tariff", "c_wr"),
    "fabs": ("R", "alpha_bar", "R_osat", "sigma_scr", "fab_hits"),
    "supply": ("supply",),
    "demand": ("demand",),
}


def frozen(marks, t: int, groups: tuple = tuple(GROUPS), straits: str = "frozen"):
    """The marks of an episode whose network stays, from week t on, as the agent of week t sees it.

    ``groups`` names the parts that are frozen; the others keep their true future. ``straits`` says what a disrupted
    strait does after this week: ``frozen`` (it stays as seen), ``open`` (it is back to normal next week), ``young`` (a
    disruption seen for at most 3 weeks is over after one more week, an older one stays).
    """
    i = t - 1
    names = {name for g in groups for name in GROUPS[g]}
    out = {}
    for name, now in NOW.items():
        if name in names:
            a = np.array(getattr(marks, name))
            a[i:] = getattr(marks, now)[i]
            out[name] = a
    for name in THIS_WEEK:
        if name in names:
            a = np.array(getattr(marks, name))
            a[i:] = a[i]
            out[name] = a
    if "demand" in names:
        d = np.array(marks.demand)
        d[i:] = d[:i].mean(axis=0) if i else d[0]
        out["demand"] = d
    if "fab_hits" in names:
        out["fab_hits"] = tuple(h for h in marks.fab_hits if h.onset_week < t)
    if "o" in names and straits != "frozen":
        seen = np.asarray(marks.o_now)
        with np.errstate(divide="ignore", invalid="ignore"):  # a strait's normal throughput: kappa / openness
            per = np.where(seen[:, :, None] > 0, np.asarray(marks.kappa_now) / seen[:, :, None], np.nan)
        normal = np.nan_to_num(np.nanmax(per, axis=0), nan=0.0)
        for c in range(seen.shape[1]):
            age = 0
            while age <= i and seen[i - age, c] < 1.0 - 1e-9:
                age += 1
            if age and (straits == "open" or age <= 3):
                back = i + 1 if straits == "open" else i + 2  # the first week it is normal again
                out["o"][back:, c] = 1.0
                out["kappa"][back:, c] = normal[c]
    return dataclasses.replace(marks, **out)


def one(agent, task, entropy, n, moves, budget, horizon, groups, straits, min_gain) -> dict:
    p = played(agent, task, entropy, n)
    inst, marks = world(task, entropy, n)
    kinds = slot_kinds(inst)
    real = Replay(inst, marks, p["actions"])  # the true episode; its flows are the plan, changed week by week
    assert real.J == p["J"], (real.J, p["J"])
    start, replays, kept, audit = time.time(), 0, 0, []
    for t in range(1, real.T + 1):
        banned = marks.prohibited[t - 1]
        plan = [dict(f) for f in real.flows]
        plan[t - 1] = {
            s: q for s, q in plan[t - 1].items() if not banned[inst.action_slots[s][0], inst.action_slots[s][1]]
        }
        seen = frozen(marks, t, groups, straits) if groups else marks
        model = Replay.from_week(inst, seen, plan, real.ov, real.holds, real.states[t - 1], t)
        before = model.J
        last = min(real.T, t + horizon - 1)
        # the same plan in the true episode, moved along with the model: what each kept move is really worth
        true = Replay.from_week(inst, marks, [dict(f) for f in plan], real.ov, real.holds, real.states[t - 1], t)

        def really(change, gain, true=true, t=t):
            J, flows = true.trial(change)
            audit.append((t, gain, true.J - J))
            true.accept(change, flows, J)

        res = search(
            model, kinds, SETS[moves], budget, 1e9, sweeps=1, last=last, by_week=True, min_gain=min_gain,
            on_accept=really,
        )  # fmt: skip
        replays += res["replays"]
        kept += model.J < before
        real.flows = model.flows
        real.J = real._run(t, real.flows, keep=True)  # week t is played for real; later weeks are still a plan
    return {"episode": n, "J": p["J"], "after": real.J, "replays": replays, "weeks_changed": kept,
            "seconds": time.time() - start, "flows": real.flows, "audit": audit}  # fmt: skip


def main(
    agent: str,
    task: str = "small",
    entropy: int = 444,
    episodes: int = 8,
    first: int = 0,
    moves: str = "fuel",
    budget: int = 600,
    horizon: int = 6,
    n_jobs: int = 4,
    truth: bool = False,
    freeze: str = "",
    straits: str = "frozen",
    min_gain: float = 0.0,
    oracle: str = "",
    out: str = "",
) -> None:
    """Play ``episodes`` episodes with the weekly search on the frozen network; the score as played and after.

    Args:
        agent: an agent folder or a name of agents/.
        task: small or full.
        entropy: the scenarios' root.
        episodes: how many episodes, starting at ``first``.
        first: the first episode index.
        moves: the move set, one of fuel, wafer, chips, fuelwafer, all.
        budget: replays per week.
        horizon: the dispatches of this many weeks ahead may change each week.
        n_jobs: workers (one episode each).
        truth: a check of the weekly loop itself: the model is the true network of every week, not the frozen one.
        min_gain: a move is kept only if the model says it saves more than this many bn USD.
        straits: a disrupted strait after this week: frozen, open or young (see ``frozen``).
        freeze: freeze only these parts (of power, edges, straits, bans, fabs, supply, demand); the rest stays true.
        oracle: a result file of ``search.py run --out`` on the same episodes, to print beside.
        out: a path to keep the results in (.pkl), optional.

    """
    from sbf_starter import scoring

    names = freeze if isinstance(freeze, tuple | list) else [g for g in str(freeze).split(",") if g]
    groups = () if truth else tuple(names) or tuple(GROUPS)
    refs = list(scoring.episode_set(task, first + episodes, entropy=entropy, n_jobs=n_jobs, verbose=False).references)
    refs = refs[first : first + episodes]
    res = Parallel(n_jobs=n_jobs)(
        delayed(one)(agent, task, entropy, n, moves, budget, horizon, groups, straits, int(min_gain * 1e11))
        for n in range(first, first + episodes)
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(pickle.dumps({"agent": agent, "task": task, "entropy": entropy, "results": res}))
    level = np.array([r["stratum"] for r in refs])
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    room = naive - np.array([r["J_clairvoyant_cents"] for r in refs], dtype=float)
    base = np.array([r["J"] for r in res], dtype=float)
    after = np.array([r["after"] for r in res], dtype=float)
    s0, s1 = pooled(level, naive - base, room), pooled(level, naive - after, room)
    print(f"{agent} on {task}, root {entropy}, episodes {first}..{first + episodes - 1}; moves {moves}")
    print(
        f"budget {budget} replays a week, horizon {horizon} weeks; {np.mean([r['seconds'] for r in res]) / 60:.1f} min"
    )
    what = (
        "true network of every week (a check)" if truth else f"frozen network ({', '.join(groups)}; straits {straits})"
    )
    print(f"as played {s0:.4f}; with the weekly search on the {what} {s1:.4f}, {s1 - s0:+.4f}")
    print(f"better in {int((after < base).sum())} of {len(res)} episodes, worse in {int((after > base).sum())}")
    a = np.array([x for r in res for x in r["audit"]], dtype=float).reshape(-1, 3)
    if len(a):  # every move the model kept: what the model said it saves and what it really saved, bn USD
        said, real_gain = a[:, 1] / 1e11, a[:, 2] / 1e11
        print(f"moves kept: {len(a) / len(res):.0f} an episode; really better: {np.mean(real_gain > 0):.0%}")
        print("   by what the model said a move saves, bn: moves, said in all, really saved in all (per episode)")
        edges = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (said > lo) & (said <= hi)
            if m.any():
                per = np.array([m.sum(), said[m].sum(), real_gain[m].sum()]) / len(res)
                print(f"   {lo:>5} .. {hi:<5}: {per[0]:6.1f}  {per[1]:8.2f}  {per[2]:8.2f}")
    known = {}
    if oracle:
        d = pickle.loads(Path(oracle).read_bytes())
        known = {r["episode"]: r["sets"][moves]["after"] for r in d["results"] if moves in r["sets"]}
        if all(r["episode"] in known for r in res):
            best = np.array([known[r["episode"]] for r in res], dtype=float)
            s2 = pooled(level, naive - best, room)
            share = (s1 - s0) / (s2 - s0) if s2 > s0 else float("nan")
            print(f"the search that knows the future: {s2:.4f}, {s2 - s0:+.4f}; the share kept without it: {share:.0%}")
    print("\nper episode (level, as played, frozen network" + (", knowing the future" if known else "") + "):")
    for i, r in enumerate(res):
        cell = f" {(naive[i] - known[r['episode']]) / room[i]:.3f}" if r["episode"] in known else ""
        mine, new = (naive[i] - base[i]) / room[i], (naive[i] - after[i]) / room[i]
        print(f"  {r['episode']:>3}  {level[i]}  {mine:.3f}  {new:.3f}{cell}")


if __name__ == "__main__":
    fire.Fire(main)
