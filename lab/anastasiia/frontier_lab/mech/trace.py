"""Play a scenario variant and keep what the week's action changed in the point plan's first week.

    uv run python lab/anastasiia/frontier_lab/mech/trace.py run two1l --episodes=64-71
    uv run python lab/anastasiia/frontier_lab/mech/trace.py show two1l

``run`` plays a variant of ``scen/play.py`` (its ``VARIANTS`` and this file's ``MORE``) and keeps, per week, the
action of the point planner (what the model alone would send in this state), the action of every scenario's planner
and the action played, as the environment's flat arrays; the slots' labels; what the environment executed; the usual
record of ``play.episode``. One file an episode in ``outputs/frontier_lab/mech/trace/<tag>_<task>_<entropy>/``,
claimed with a lock file as ``play.run`` does. Such a play is the variant's own play: its cost is the kept one's on
the same machine.

``show`` sums, over the weeks the joint first week was played, what it asks more and less than the point plan, by
the kind of slot (fuel order, valve into a grid, wafer, raw chip, packaged chip, tanker release).
"""

import os
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE.parent / "scen"), str(ROOT / "lab" / "anastasiia" / "hazard_lab")]
OUT = ROOT / "outputs" / "frontier_lab" / "mech" / "trace"


def labels(agent) -> dict:
    """The slots of the action as an analysis reads them."""
    m, inst = agent.model, agent.model.inst
    name = [c.id for c in inst.commodities]
    slots = []
    for s, (e, k, lane) in enumerate(inst.action_slots):
        edge = inst.edges[e]
        last = edge if lane is None else inst.edges[inst.lanes[lane].edges[-1]]
        slots.append({"kind": agent.kinds[s], "k": name[k], "edge": edge.id, "tail": inst.nodes[edge.tail].id,
                      "head": inst.nodes[edge.head].id, "to": inst.nodes[last.head].id, "lane": lane, "tau": edge.tau,
                      "u0": edge.u0, "e": int(e)})  # fmt: skip
    releases = []
    for o, (c, k, e, lane) in enumerate(inst.override_slots):
        releases.append({"strait": inst.nodes[c].id, "k": name[k], "edge": inst.edges[e].id, "pair": int(m.ov_pair[o])})
    return {"slots": slots, "releases": releases, "pairs": [(inst.nodes[c].id, name[k]) for c, k in m.pairs],
            "grids": [inst.nodes[g].id for g in inst.grids], "fabs": [inst.nodes[f].id for f in inst.fabs],
            "stock": [(inst.nodes[sl.node].id, name[sl.k]) for sl in inst.stock_slots]}  # fmt: skip


def episode(tag: str, task: str, entropy: int, n: int, weeks: int = 0) -> dict:
    import gymnasium as gym
    import play as P
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    import world as W
    from planner import Scen
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    opts = dict(P.VARIANTS[tag])
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    t0 = time.process_time()
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    kw = {key: opts.pop(key) for key in getattr(P, "WORLD_KEYS", ("new", "ends", "ages")) if key in opts}
    world = P._world(task, u, n, max(opts.get("K", 0), opts.get("expect", 0)), info["policy_seed"], **kw)
    base = opts.pop("base", P.BASE[task])
    Agent = load(str(resolve(str(ROOT / base)).resolve()))
    ag = Scen(Agent, config, world, groups=W.GROUPS, truth=u.core._ep.marks, **opts)
    seen: list = []  # the week's actions of the point planner and of the scenarios' planners, in that order

    def keep(planner):
        act = planner.act

        def wrapped(observation):
            a = act(observation)
            seen.append({k: np.array(a[k]) for k in ("flows", "override_qty", "release_mode")})
            return a

        planner.act = wrapped

    for planner in [ag.point] + ag.scen:
        keep(planner)
    cpu, done, trace = [], False, []
    while not done:
        seen.clear()
        action = ag.act(obs)
        cpu.append(time.process_time() - t0)
        trace.append({"point": seen[0], "scen": seen[1:], "final": {k: np.array(action[k]) for k in seen[0]},
                      "stock": np.array(obs["stock.qty"], dtype=float),
                      "u": np.array(obs["graph_now.u"], dtype=float), "open": np.array(obs["graph_now.open"], dtype=float),
                      "G_bar": np.array(obs["graph_now.grid.G_bar"], dtype=float),
                      "prohibited": np.array(obs["graph_now.prohibited"]).astype(bool)})
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc or (weeks and len(cpu) >= weeks)
        t0 = time.process_time()
    recs = u.core._ep.traj.records
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    inst = u.core._ep.inst
    sent = np.zeros((len(recs), len(inst.action_slots)))
    for t, r in enumerate(recs):
        for slot, q in r.executed.items():
            sent[t, slot] = q
    ag.notes["redrawn"] = world.redrawn
    return {
        "n": n, "J": int(sum(r.cost_cents for r in recs) if weeks else u.core._ep.traj.J_cents), "cpu": cpu, "log": list(ag.log), "notes": ag.notes,
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "weeks": len(recs), "sent": sent,
        "point_log": list(ag.point.log), "trace": trace, "labels": labels(ag.point),
    }  # fmt: skip


def run(*tags: str, task: str = "small", entropy: int = 444, episodes="64-71", weeks: int = 0) -> None:
    import play as P

    for tag in tags:
        folder = OUT / f"{tag}_{task}_{entropy}"
        folder.mkdir(parents=True, exist_ok=True)
        for n in P._numbers(episodes):
            done, lock = folder / f"{n}.pkl", folder / f"{n}.lock"
            if not weeks:
                if done.is_file():
                    continue
                try:
                    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
            started = time.time()
            r = episode(tag, task, entropy, n, weeks)
            if weeks:
                print(tag, n, "J", r["J"], "cpu", round(sum(r["cpu"]), 1), r["notes"]["chosen"], flush=True)
                continue
            tmp = folder / f"{n}.tmp"
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(done)
            lock.unlink(missing_ok=True)
            print(f"{tag} {task} {entropy} episode {n}: {time.time() - started:.0f} s wall, {sum(r['cpu']):.0f} s CPU, "
                  f"{r['notes']['chosen']}", flush=True)  # fmt: skip


def kept(tag: str, task: str = "small", entropy: int = 444) -> dict:
    folder = OUT / f"{tag}_{task}_{entropy}"
    return {int(p.stem): pickle.loads(p.read_bytes()) for p in folder.glob("*.pkl")} if folder.is_dir() else {}


def show(tag: str, task: str = "small", entropy: int = 444, top: int = 12) -> None:
    data = kept(tag, task, entropy)
    if not data:
        print("nothing kept")
        return
    lab = next(iter(data.values()))["labels"]
    kinds = np.array([s["kind"] for s in lab["slots"]])
    more, less, base, weeks, played = {}, {}, {}, 0, 0
    by_slot = np.zeros((len(kinds), 3))  # more, less, the point plan's
    rel = np.zeros(3)
    for r in data.values():
        for w, row in zip(r["trace"], r["notes"]["rows"]):
            weeks += 1
            if row["pick"] != "joint":
                continue
            played += 1
            a, b = w["point"]["flows"], w["final"]["flows"]
            d = b - a
            by_slot += np.stack([np.maximum(d, 0), np.maximum(-d, 0), a], axis=1)
            ra = np.where(w["point"]["release_mode"][[x["pair"] for x in lab["releases"]]] == 1, w["point"]["override_qty"], 0.0)
            rb = np.where(w["final"]["release_mode"][[x["pair"] for x in lab["releases"]]] == 1, w["final"]["override_qty"], 0.0)
            rel += (np.maximum(rb - ra, 0).sum(), np.maximum(ra - rb, 0).sum(), ra.sum())
    for kind in sorted(set(kinds)):
        m = kinds == kind
        more[kind], less[kind], base[kind] = by_slot[m, 0].sum(), by_slot[m, 1].sum(), by_slot[m, 2].sum()
    print(f"{tag} {task} {entropy}: {len(data)} episodes, {weeks} weeks, the joint first week played in {played}")
    print("  requests of the joint first week against the point plan's, as shares of the point plan's of the same kind:")
    for kind in sorted(set(kinds)):
        print(f"    {kind:6s} more {more[kind] / base[kind]:+.3f}  less {-less[kind] / base[kind]:+.3f}  net "
              f"{(more[kind] - less[kind]) / base[kind]:+.4f}")
    print(f"    release more {rel[0] / max(rel[2], 1e-9):+.3f}  less {-rel[1] / max(rel[2], 1e-9):+.3f}  net {(rel[0] - rel[1]) / max(rel[2], 1e-9):+.4f}")
    order = np.argsort(-(by_slot[:, 0] + by_slot[:, 1]) / np.maximum(by_slot[:, 2].sum(), 1e-9) * 0 - (by_slot[:, 0] + by_slot[:, 1]))
    print("  the slots that move most (units a played week: more, less, the point plan's):")
    for s in order[:top]:
        x = lab["slots"][s]
        print(f"    {x['kind']:6s} {x['k']:12s} {x['tail']:>16s} -> {x['to']:16s} {by_slot[s, 0] / played:10.1f} {by_slot[s, 1] / played:10.1f} {by_slot[s, 2] / played:10.1f}")


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
