"""Worker LS: local search over the played actions of planner_H's full-episode plan, the simulator as the evaluator.

    # 1. solve planner_H's MILP once per episode and cache its actions (outputs/planner_LS/cache/ep<n>.pkl)
    uv run python lab/anastasiia/mpc_lab/planners/planner_LS.py prepare --episodes=4 --n_jobs=2
    # 2. local search on the cached actions; prints per-episode claimed / played-before / played-after and RSS
    uv run python lab/anastasiia/mpc_lab/planners/planner_LS.py search --episodes=4 --budget=800 --n_jobs=2

Everything is known (open loop, one episode's omega), root 111, task small.

Replay. The plan's weekly wire actions (``lp_common.week1_action`` of each week's plan columns, exactly as
``planner_H.play`` builds them) are validated once with the Env's own ``validate_action`` and then played with the raw
simulator step (``dynamics.sim.step``), skipping the observation. The state at the end of every week of the incumbent
is kept (deep copies), so a move that changes week t replays only weeks t..T. J = sum of weekly cost cents minus the
terminal salvage cents, as ``Trajectory.J_cents``; ``check`` asserts that it equals the Env's played J.

Moves (on action slots only; overrides / holds at chokepoints stay as the plan has them):
  shift   move a fraction (1, 1/2) of a dispatch (slot, week) to week +-1 / +-2 of the same slot
  scale   multiply a dispatch by 0, 0.5, 1.5
  merge   = shift with fraction 1 (the whole dispatch joins the neighbouring week's dispatch on the same slot)
A move whose target (slot, week) is not a valid entry of that week (masked, prohibited) is skipped: the validator
must drop nothing new. Only strict improvements of the played J are accepted (first improvement).
"""

import copy
import os
import pickle
import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
# the team's committed cache (F_Q quantiles, references), as sbf_starter sets it; without it a fresh container spends
# many minutes per process recomputing the F_Q quantiles inside planner_H.world
os.environ.setdefault("SBF_CACHE_DIR", os.path.join(ROOT, "hub", "refcache"))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import planner_H as H  # noqa: E402
from package_baselines import rss  # noqa: E402


def world_nofq(n):
    """planner_H.world without the naive fallback: (instance, omega, marks, None).

    ``_world`` also loads naive's F_Q quantiles, which only the fallback needs; the small task's entry is not in the
    team cache under this package's key, and computing it takes tens of minutes on a loaded machine. The plan's actions
    are always well formed (no whole-week failure), so the fallback never plays and the played J is the same.
    """
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow_agent.scoring import _label

    inst, params = task_generator(H.TASK)
    omega = sample_omega(inst, params, H.ENTROPY, n, _label(H.ENTROPY))
    return inst, omega, compute_marks(inst, omega), None


def patch():
    """Install world_nofq in planner_H; called in every worker too (loky workers import planner_H afresh)."""
    H.world = world_nofq


patch()

CACHE = os.path.join(ROOT, "outputs", "planner_LS", "cache")


# ----- the plan's actions ---------------------------------------------------------------------------------------------
def plan_actions(n, m, x):
    """The weekly wire actions of the plan, as ``planner_H.play`` builds them, and the Env's played J."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = H.world(n)
    nc = m.meta["nc"]
    env = Env(fallback=fb)
    obs, info = env.reset(inst, "standard", omega, _policy_seed(H.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="planLS")
    done, t, acts = False, 0, []
    while not done:
        xt = np.zeros_like(m.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        a = L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t]))
        acts.append(copy.deepcopy(a))
        obs, r, done, tr, inf = env.step(a)
        t += 1
    return acts, int(env.trajectory.J_cents), env.trajectory.records


def plan_columns(inst, m, x):
    """A few plan series for the diagnostics: lots (T, F), wafer stock at fabs (T, F), shed (T, G)."""
    nc, T, tm = m.meta["nc"], m.T, m.meta["template"]
    c = lambda tag, t, *r: float(x[(t - 1) * nc + tm[(tag, *r)]]) if (tag, *r) in tm else 0.0
    wslots = [inst.slot_index[(f, inst.nodes[f].fab.input)] for f in inst.fabs]
    return dict(
        lots=np.array([[c("p", t, f) for f in range(len(inst.fabs))] for t in range(1, T + 1)]),
        wafers=np.array([[c("I", t, s) for s in wslots] for t in range(1, T + 1)]),
        shed=np.array([[c("ysh", t, g) for g in range(len(inst.grids))] for t in range(1, T + 1)]),
    )


def solve_lazy_keep(n, rules, tl=60, gap=1e-3, rounds=8):
    """planner_H.solve_lazy, except that a round with no solution in the time limit ends the loop with the previous
    round's plan instead of failing (episode 2: round 2 found no incumbent in 60 s)."""
    from shockbench_flow.dynamics.sim import initial_stock

    w = H.world(n)
    inst = w[0]
    i0 = initial_stock(inst)
    gw, ow, pref, last, hist = set(), set(), None, None, []
    for it in range(rounds + 1):
        out, sol = H.solve_full(n, rules, tl, gap, only_gw=gw, only_ow=ow, world_=w, phat_ref=pref)
        if sol is None:
            print(f"  ep {n} round {it}: no solution in {tl}s, keeping round {it - 1}", file=sys.stderr, flush=True)
            break
        last = (out, sol)
        m, x = sol
        g2, o2, kinds, pref = H.violations(inst, m, x, i0)
        hist.append((it, len(gw), len(ow), len(g2 - gw), len(o2 - ow), round(out["secs"]), out["gap"], out["J"]))
        print(f"  ep {n} round {it}: binaries on {len(gw)} grid-weeks {len(ow)} osat-weeks; violations {len(g2)} / {len(o2)} {kinds} gap {out['gap']} J {out['J']} {out['secs']:.0f}s", file=sys.stderr, flush=True)
        if not (g2 - gw) and not (o2 - ow):
            break
        gw |= g2
        ow |= o2
    if last is None:
        return {"J": None}, None
    out, sol = last
    out["hist"], out["rounds"], out["flagged"] = hist, len(hist) - 1, (len(gw), len(ow))
    return out, sol


def prepare_one(n, tl, gap, rules, rounds=8):
    patch()
    path = os.path.join(CACHE, f"ep{n}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return n, d["claimed"], d["played"], d["secs"]
    t0 = time.time()
    out, sol = H.solve_lazy(n, H.parse(rules), tl=tl, gap=gap, rounds=rounds, log=True)
    if sol is None:
        out, sol = solve_lazy_keep(n, H.parse(rules), tl=tl, gap=gap, rounds=rounds)
    if sol is None:
        return n, None, None, time.time() - t0
    m, x = sol
    acts, J, R = plan_actions(n, m, x)
    J_h, _ = H.play(n, m, x)
    assert J == J_h, (J, J_h)
    inst = H.world(n)[0]
    d = dict(n=n, claimed=out["J"], played=J, actions=acts, plan=plan_columns(inst, m, x), gap=out.get("gap"),
             secs=time.time() - t0, rounds=out.get("rounds"), flagged=out.get("flagged"))
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(d, f)
    os.replace(path + ".tmp", path)
    return n, out["J"], J, d["secs"]


def prepare(episodes=4, start=0, tl=60, gap=1e-3, rules="base,fuel,lots", rounds=8, n_jobs=2):
    res = Parallel(n_jobs=n_jobs)(delayed(prepare_one)(n, tl, gap, rules, rounds) for n in range(start, start + episodes))
    for n, c, p, s in res:
        print(f"ep {n}: claimed J {c} played J {p} ({s:.0f}s)", flush=True)


def load(n):
    with open(os.path.join(CACHE, f"ep{n}.pkl"), "rb") as f:
        return pickle.load(f)


# ----- the fast replay ------------------------------------------------------------------------------------------------
class Replay:
    """The episode played from validated weekly actions with the raw simulator step, with checkpoints per week."""

    def __init__(self, n, actions):
        from shockbench_flow.dynamics.env import validate_action
        from shockbench_flow.dynamics.sim import initial_state

        inst, omega, marks, fb = H.world(n)
        self.inst = inst.at_digest(marks.instance_digest)
        self.marks = marks
        self.T = self.inst.T
        self.flows, self.ov, self.holds = [], [], []
        for t, a in enumerate(actions, start=1):
            fl, ov, ho, inv = validate_action(self.inst, marks, t, a)
            if inv:
                raise ValueError(f"week {t}: the plan's action has invalid entries {inv[:3]}")
            self.flows.append(dict(fl)), self.ov.append(ov), self.holds.append(ho)
        s0 = initial_state(self.inst)
        s0.last = None
        self.states = [s0] + [None] * self.T  # states[t]: end of week t (last=None)
        self.costs = [0] * (self.T + 1)  # costs[t]: cost cents of week t
        self.records = [None] * (self.T + 1)
        self.salvage = None
        self.calls = 0
        self.weeks = 0
        self.J = self.commit(1)

    def _run(self, t0, flows, keep):
        """Play weeks t0..T from states[t0 - 1] with ``flows`` (list by week, 0-based). Returns J cents."""
        from shockbench_flow.dynamics.sim import step, terminal_salvage
        from shockbench_flow.dynamics.state import cents

        st = copy.deepcopy(self.states[t0 - 1])
        J = sum(self.costs[1:t0])
        self.calls += 1
        for t in range(t0, self.T + 1):
            rec = step(self.inst, self.marks, st, flows[t - 1], self.ov[t - 1], self.holds[t - 1])
            self.weeks += 1
            J += rec.cost_cents
            if keep:
                self.costs[t], self.records[t] = rec.cost_cents, rec
                st.last = None
                if t < self.T:
                    self.states[t] = copy.deepcopy(st)
        S = cents(terminal_salvage(self.inst, st))
        if keep:
            self.salvage = S
        return J - S

    def commit(self, t0):
        self.J = self._run(t0, self.flows, keep=True)
        return self.J

    def trial(self, changes):
        """J of the incumbent with ``changes`` {(week, slot): qty} applied (nothing kept)."""
        t0 = min(t for t, _ in changes)
        flows = list(self.flows)
        for (t, s), q in changes.items():
            if t - 1 < len(flows) and flows[t - 1] is self.flows[t - 1]:
                flows[t - 1] = dict(self.flows[t - 1])
            flows[t - 1][s] = q
        return self._run(t0, flows, keep=False), flows

    def accept(self, changes, flows, J):
        self.flows = flows
        t0 = min(t for t, _ in changes)
        J2 = self.commit(t0)
        assert J2 == J, (J2, J)


def check(n=0):
    """The fast replay reproduces the Env's played J."""
    d = load(n)
    t0 = time.time()
    R = Replay(n, d["actions"])
    t1 = time.time()
    print(f"ep {n}: Env played J {d['played']}  replay J {R.J}  equal {R.J == d['played']}  ({t1 - t0:.2f}s with setup)")
    t0 = time.process_time()
    for _ in range(5):
        R._run(1, R.flows, keep=False)
    print(f"full replay {((time.process_time() - t0) / 5):.3f} cpu-s; from week 27: ", end="")
    t0 = time.process_time()
    for _ in range(5):
        R._run(27, R.flows, keep=False)
    print(f"{((time.process_time() - t0) / 5):.3f} cpu-s")


# ----- local search ---------------------------------------------------------------------------------------------------
def slot_kinds(inst):
    """Per action slot: 'wafer' (ends at a fab, its wafer input), 'fuel' (ends at a grid, one of its fuels), 'other'."""
    out = []
    for e, k, lane in inst.action_slots:
        dest = inst.edges[inst.lanes[lane].edges[-1]].head if lane is not None else inst.edges[e].head
        nd = inst.nodes[dest]
        if nd.fab is not None and k == nd.fab.input:
            out.append("wafer")
        elif nd.grid is not None and k in nd.grid.fuels:
            out.append("fuel")
        else:
            out.append("other")
    return out


MOVES = (  # (name, target week offset or None for a scale, fraction moved / scale factor)
    ("x1.5", None, 1.5),
    ("drop", None, 0.0),
    ("shift-1", -1, 1.0),
    ("shift+1", 1, 1.0),
    ("half", None, 0.5),
    ("x2", None, 2.0),
    ("half-1", -1, 0.5),
    ("half+1", 1, 0.5),
    ("shift+2", 2, 1.0),
    ("shift-2", -2, 1.0),
)


def moves_of(R, t, s):
    q = R.flows[t - 1].get(s, 0.0)
    if q <= 0:
        return
    for name, dt, f in MOVES:
        if dt is None:
            yield name, {(t, s): q * f}
        else:
            t2 = t + dt
            if not 1 <= t2 <= R.T or s not in R.flows[t2 - 1]:  # the target entry must be valid in that week
                continue
            yield name, {(t, s): q * (1.0 - f), (t2, s): R.flows[t2 - 1][s] + q * f}


def candidates(R, kinds, order):
    """(t, s) dispatches of the incumbent, ranked: wafer and fuel slots first, then by customs value of the dispatch."""
    v = np.array([R.inst.commodities[k].v for _, k, _ in R.inst.action_slots])
    rank = {"wafer": 0, "fuel": 1, "other": 2} if order == "kind" else {"wafer": 0, "fuel": 0, "other": 0}
    c = [(rank[kinds[s]], -q * v[s], t, s) for t in range(1, R.T + 1) for s, q in R.flows[t - 1].items() if q > 1e-9]
    c.sort()
    return [(t, s) for _, _, t, s in c]


def search_one(n, budget=800, secs=600, order="kind", sweeps=3, repeat=3):
    patch()
    d = load(n)
    t0 = time.time()
    c0 = time.process_time()
    R = Replay(n, d["actions"])
    assert R.J == d["played"], (R.J, d["played"])
    J0 = R.J
    kinds = slot_kinds(R.inst)
    stats = {}  # move name -> [tried, accepted, gain cents]
    bykind = {}
    acc = 0
    for sweep in range(sweeps):
        improved = False
        for t, s in candidates(R, kinds, order):
            if R.calls >= budget or time.time() - t0 > secs:
                break
            for _ in range(repeat):  # after an improvement, the same dispatch is tried again (e.g. x1.5 twice)
                hit = False
                for name, ch in moves_of(R, t, s):
                    if R.calls >= budget or time.time() - t0 > secs:
                        break
                    J, flows = R.trial(ch)
                    st = stats.setdefault(name, [0, 0, 0])
                    st[0] += 1
                    if J < R.J:
                        gain = R.J - J
                        R.accept(ch, flows, J)
                        st[1] += 1
                        st[2] += gain
                        bk = bykind.setdefault(kinds[s], [0, 0])
                        bk[0] += 1
                        bk[1] += gain
                        acc += 1
                        improved = hit = True
                        break  # first improvement
                if not hit:
                    break
        if not improved or R.calls >= budget or time.time() - t0 > secs:
            break
    return dict(n=n, claimed=d["claimed"], before=J0, after=R.J, accepted=acc, replays=R.calls, weeks=R.weeks,
                secs=time.time() - t0, cpu=time.process_time() - c0, stats=stats, bykind=bykind, sweeps=sweep + 1,
                kinds={k: kinds.count(k) for k in set(kinds)},
                ndisp=sum(1 for t in range(R.T) for q in R.flows[t].values() if q > 1e-9))


# harm cut points (USD) of the small generator, read off root 222: the largest harm of level s and the smallest of
# level s + 1 in hub/eval/records (levels) and hub/refcache (harm). The package's own cut points
# (cut_points_cached) and F_Q quantiles miss the team cache in this container and take tens of minutes to compute, so
# scoring.episode_set cannot be called here; an episode whose harm falls inside a gap is refused.
GAPS = ((1.642365690340e12, 1.657138570985e12), (2.157749339795e12, 2.206615307799e12), (2.731010041717e12, 2.899686292519e12))


def references(ns):
    import glob
    import json

    base = glob.glob(os.path.join(ROOT, "hub", "refcache", "references", "*", H.TASK, "*", f"entropy-{H.ENTROPY}", "fq-1000-*"))
    # this container's instance differs from the team's (see reports/research_full_and_ceiling.md), so two instance
    # directories can coexist: keep those that hold every requested episode, newest first (the container's own)
    base = sorted(
        (b for b in base if all(os.path.exists(os.path.join(b, f"{n}.json")) for n in ns)),
        key=os.path.getmtime,
        reverse=True,
    )
    assert base, "no reference directory holds all requested episodes"
    rows = []
    for n in ns:
        with open(os.path.join(base[0], f"{n}.json")) as f:
            r = json.load(f)
        h = r["harm_usd"]
        if any(lo <= h <= hi for lo, hi in GAPS):
            raise ValueError(f"episode {n}: harm {h:.4g} falls between two levels' observed ranges")
        r["stratum"] = 1 + sum(h > hi for lo, hi in GAPS)
        rows.append(r)
    return rows


def search(episodes=4, start=0, budget=800, secs=600, order="kind", sweeps=3, repeat=3, n_jobs=2):
    refs = references(range(start, start + episodes))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(search_one)(n, budget, secs, order, sweeps, repeat) for n in range(start, start + episodes))
    print(f"local search: budget {budget} replays, {secs}s cap, order={order}, sweeps<={sweeps}, repeat {repeat}  ({time.time() - t0:.0f}s wall)")
    print(f"{'ep':>3} {'claimed J':>16} {'played before':>16} {'played after':>16} {'gain %':>7} {'acc':>4} {'replays':>7} {'weeks':>7} {'secs':>6} {'cpu':>6} {'rep/s':>6}")
    for r in res:
        print(f"{r['n']:>3} {r['claimed']:>16} {r['before']:>16} {r['after']:>16} {100 * (r['before'] - r['after']) / r['before']:>7.3f} "
              f"{r['accepted']:>4} {r['replays']:>7} {r['weeks']:>7} {r['secs']:>6.0f} {r['cpu']:>6.0f} {r['replays'] / r['cpu']:>6.1f}")
    for r in res:
        print(f"  ep {r['n']}: dispatches {r['ndisp']} slots {r['kinds']} sweeps {r['sweeps']}; moves "
              + ", ".join(f"{k} {v[1]}/{v[0]} {v[2] / 100:.3g}$" for k, v in r["stats"].items())
              + "; by kind " + ", ".join(f"{k} {v[0]} {v[1] / 100:.3g}$" for k, v in r["bykind"].items()))
    for name, key in (("claimed", "claimed"), ("played before", "before"), ("played after", "after")):
        sc, lv = rss(refs, [r[key] for r in res])
        print(f"RSS {name:14s} {sc:.4f}  per episode " + " ".join(f"{rss([refs[i]], [r[key]])[0]:.4f}" for i, r in enumerate(res)), flush=True)


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "check": check, "search": search})
