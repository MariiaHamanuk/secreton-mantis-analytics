"""Worker LSF: fuel-only local search on the played trajectory of agents/anastasiia_hybrid_chiplp, the simulator as judge.

    uv run python lab/anastasiia/mpc_lab/planners/planner_LSF.py run --episodes=12 --budget=20000 --secs=1200 --n_jobs=2
    uv run python lab/anastasiia/mpc_lab/planners/planner_LSF.py verify --n=0     # play + replay check only

1. Play. The hybrid agent plays episode n of root ``ENTROPY`` through gymnasium exactly as hub/eval/diag.py does
   (``gym.make(env_id(task), entropy=E)``, ``reset(options={"episode": n})``, ``sbf_starter.agents.load``; policy seed
   0, no fallback). Each week's raw action dict (flows, override_qty, release_mode) is kept, with the episode's J and the
   wire actions the core Env stored (outputs/planner_LSF/cache/ep<n>.pkl).
2. Convert. Each week's arrays become the wire action (``to_wire``): flows = the nonzero slots; overrides = every
   override slot of a release pair in mode 1 (static["override_slots"] -> layout["release_pairs"]); hold = the pairs in
   mode 2. ``to_wire`` is checked against the package's ``action_from_flat`` and the Env's stored actions (after
   ``validate_action``), and the fast replay (planner_LS.Replay: raw ``dynamics.sim.step`` from cached weekly states)
   must give the gym episode's J to the cent before any search.
3. Search. planner_LS's moves (scale x1.5 / x0 / x0.5 / x2, shift the whole or half of a dispatch by +-1 / +-2 weeks)
   with strict first-improvement acceptance, on FUEL action slots only (lng, crude, nucfuel): upstream orders (source ->
   strait / terminal / grid) and the zero-lag terminal -> grid valves. Candidates: fuel dispatches ranked by qty x v_k.
   Overrides and holds at the straits stay as the agent played them (v1 limitation). Unlike planner_LS, a shift may
   land on a week where the slot sent nothing, as long as the slot is not prohibited that week (validate_action would
   drop it); a valve that is closed in the agent's play is often exactly where the fuel should go.

Everything is known (open loop, one episode's omega): the result is a feasible, simulator-verified schedule that bounds
what better fuel timing could gain on top of the hybrid agent, not a policy.
"""

import collections
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
os.environ.setdefault("SBF_CACHE_DIR", os.path.join(ROOT, "hub", "refcache"))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import planner_LS as LS  # noqa: E402

TASK, ENTROPY = "small", 444  # defaults; configure() switches them (also inside every joblib worker)
AGENT = os.path.join(ROOT, "agents", "anastasiia_hybrid_chiplp")
CACHE = os.path.join(ROOT, "outputs", "planner_LSF", "cache")
LOG = None  # optional progress log path (appended to, flushed per line)
FUELS = ("lng", "crude", "nucfuel")
OVERRIDE, HOLD = 1, 2  # release_mode codes


def configure(task=None, entropy=None, log=None):
    """Set the module task / root. small@444 keeps the original cache dir; other pairs get their own."""
    global TASK, ENTROPY, CACHE, LOG
    TASK = task or TASK
    ENTROPY = int(entropy) if entropy is not None else ENTROPY
    sub = "cache" if (TASK, ENTROPY) == ("small", 444) else f"cache_{TASK}_{ENTROPY}"
    CACHE = os.path.join(ROOT, "outputs", "planner_LSF", sub)
    if log:
        LOG = log


def logline(msg):
    print(msg, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(msg + "\n")


def world(n):
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.marks import compute_marks

    inst, _params = task_generator(TASK)
    omega = scenario(TASK, n, entropy=ENTROPY)
    marks = compute_marks(inst, omega)
    return inst.at_digest(marks.instance_digest), omega, marks


# ----- 1. the agent's played trajectory -------------------------------------------------------------------------------
def play(n):
    """Play the hybrid agent on episode n as diag.py does; cache raw actions, J and the Env's stored wire actions."""
    path = os.path.join(CACHE, f"ep{n}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    t0 = time.time()
    env = gym.make(env_id(TASK), entropy=ENTROPY)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    agent = load(AGENT)(config)
    raw, rewards, done = [], 0, False
    while not done:
        a = agent.act(obs)
        raw.append({k: np.array(v, copy=True) for k, v in a.items()})
        obs, _r, term, trunc, inf = env.step(a)
        rewards += inf["reward_cents"]
        done = term or trunc
    traj = u.core.trajectory
    d = dict(
        n=n,
        raw=raw,
        J=int(traj.J_cents),
        J_rewards=-int(rewards),
        wire=copy.deepcopy(traj.actions),
        policy_seed=info["policy_seed"],
        release_pairs=[tuple(map(int, p)) for p in config["layout"]["release_pairs"]],
        override_slots={k: list(map(int, v)) for k, v in config["static"]["override_slots"].items() if k in ("chokepoint", "k")},
        layout=u.layout,
        omega_hash=traj.omega_hash,
        secs=time.time() - t0,
        fallback_weeks=sum(1 for r in traj.records if r.invalid and str(r.invalid[0]).startswith("whole-week")),
    )
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(d, f)
    os.replace(path + ".tmp", path)
    return d


# ----- 2. arrays -> wire action ---------------------------------------------------------------------------------------
def to_wire(week, a, release_pairs, override_slots):
    """The wire action of week ``week`` from the raw arrays: the inverse of the agents' assembly of their return."""
    flows = np.asarray(a["flows"], dtype=float)
    slots = [s for s in range(len(flows)) if flows[s] != 0]
    act = {"week": week, "flows": {"slot": slots, "qty": [float(flows[s]) for s in slots]}}
    modes = np.asarray(a.get("release_mode", np.zeros(len(release_pairs), dtype=int)))
    oq = np.asarray(a.get("override_qty", np.zeros(len(override_slots["k"]))), dtype=float)
    pair = {p: i for i, p in enumerate(release_pairs)}
    opair = [pair[(c, k)] for c, k in zip(override_slots["chokepoint"], override_slots["k"])]
    ov = [o for o, p in enumerate(opair) if modes[p] == OVERRIDE]
    if ov:
        act["overrides"] = {"slot": ov, "qty": [float(oq[o]) for o in ov]}
    held = [release_pairs[p] for p in range(len(release_pairs)) if modes[p] == HOLD]
    if held:
        act["hold"] = {"chokepoint": [c for c, _ in held], "k": [k for _, k in held]}
    return act


def wire_actions(d, inst, marks):
    """Wire actions of the recorded episode, checked against action_from_flat and the Env's stored actions."""
    from shockbench_flow.dynamics.env import validate_action
    from shockbench_flow.information.flat import action_from_flat

    out, bad = [], []
    for t, a in enumerate(d["raw"], start=1):
        w = to_wire(t, a, d["release_pairs"], d["override_slots"])
        ref = action_from_flat(d["layout"], t, a["flows"], a.get("override_qty", np.zeros(d["layout"].n_override_slots)),
                               a.get("release_mode", np.zeros(len(d["layout"].pairs), dtype=np.int64)))
        v_mine, v_ref, v_env = (validate_action(inst, marks, t, x) for x in (w, ref, d["wire"][t - 1]))
        if not (v_mine[:3] == v_ref[:3] == v_env[:3]):
            bad.append(t)
        out.append(w)
    return out, bad


class FReplay(LS.Replay):
    """planner_LS.Replay on given (instance, marks), dropping the entries the validator drops (as the Env does)."""

    def __init__(self, inst, marks, actions):
        from shockbench_flow.dynamics.env import validate_action
        from shockbench_flow.dynamics.sim import initial_state

        self.inst, self.marks, self.T = inst, marks, inst.T
        self.flows, self.ov, self.holds, self.dropped = [], [], [], 0
        for t, a in enumerate(actions, start=1):
            fl, ov, ho, inv = validate_action(inst, marks, t, a)
            self.dropped += len(inv)
            self.flows.append(dict(fl)), self.ov.append(ov), self.holds.append(ho)
        s0 = initial_state(inst)
        s0.last = None
        self.states = [s0] + [None] * self.T
        self.costs = [0] * (self.T + 1)
        self.records = [None] * (self.T + 1)
        self.salvage = None
        self.calls = self.weeks = 0
        self.J = self.commit(1)

    def components(self):
        from shockbench_flow.dynamics.state import COST_COMPONENTS

        tot = dict.fromkeys(COST_COMPONENTS, 0.0)
        for r in self.records[1:]:
            for c, v in r.costs.as_dict().items():
                tot[c] += v
        tot["salvage"] = -self.salvage / 100
        return tot


def verify_one(n):
    """Play (or load) episode n and check the fast replay; (d, R, report)."""
    d = play(n)
    inst, _omega, marks = world(n)
    acts, bad = wire_actions(d, inst, marks)
    R = FReplay(inst, marks, acts)
    rep = dict(n=n, gym_J=d["J"], rewards_J=d["J_rewards"], replay_J=R.J, equal=R.J == d["J"], bad_weeks=bad,
               dropped=R.dropped, fallback_weeks=d["fallback_weeks"], play_secs=d["secs"], omega_hash=d["omega_hash"])
    return d, R, rep


def verify(n=0, task="small", entropy=444):
    configure(task, entropy)
    _d, _R, rep = verify_one(n)
    print(rep, flush=True)


# ----- 3. fuel-only local search --------------------------------------------------------------------------------------
def fuel_slots(inst):
    """{slot: (commodity name, 'valve' | 'order')} of the fuel action slots."""
    out = {}
    for s, (e, k, _lane) in enumerate(inst.action_slots):
        name = inst.commodities[k].id
        if name in FUELS:
            ed = inst.edges[e]
            valve = inst.nodes[ed.tail].terminal is not None and inst.nodes[ed.head].grid is not None
            out[s] = (name, "valve" if valve else "order")
    return out


def slot_name(inst, s):
    e, k, lane = inst.action_slots[s]
    return inst.edges[e].id + ("" if lane is None else f" [{inst.lanes[lane].id}]")


def moves_of(R, t, s):
    """planner_LS.MOVES on (t, s); a shift target must be a week where the slot is not prohibited (may hold 0)."""
    q = R.flows[t - 1].get(s, 0.0)
    if q <= 0:
        return
    e, k, _ = R.inst.action_slots[s]
    for name, dt, f in LS.MOVES:
        if dt is None:
            yield name, {(t, s): q * f}
        else:
            t2 = t + dt
            if not 1 <= t2 <= R.T or R.marks.prohibited[t2 - 1][e, k]:
                continue
            yield name, {(t, s): q * (1.0 - f), (t2, s): R.flows[t2 - 1].get(s, 0.0) + q * f}


def bands():
    """Week bands: the original three for Small (52 weeks), quarters of the year for Full (104 weeks)."""
    return ["1-13", "14-39", "40-52"] if TASK == "small" else ["1-26", "27-52", "53-78", "79-104"]


def band(t):
    if TASK == "small":
        return "1-13" if t <= 13 else ("14-39" if t <= 39 else "40-52")
    return bands()[min((t - 1) // 26, 3)]


def search_one(n, budget, secs, sweeps=3, repeat=3, task=None, entropy=None, log=None):
    configure(task, entropy, log)
    done = os.path.join(CACHE, f"lsf_ep{n}.pkl")
    if os.path.exists(done):  # resume: a finished episode is not searched again
        with open(done, "rb") as f:
            out = pickle.load(f)
        out.pop("flows", None)
        logline(f"[ep {n}] loaded finished search: dJ {(out['before'] - out['after']) / 1e11:.3f} bn")
        return out
    t0, c0 = time.time(), time.process_time()
    d, R, rep = verify_one(n)
    logline(f"[ep {n}] verify: gym J {rep['gym_J']} replay J {rep['replay_J']} equal {rep['equal']} bad weeks {rep['bad_weeks']} "
            f"dropped {rep['dropped']} fallback weeks {rep['fallback_weeks']} play {rep['play_secs']:.0f}s")
    out = dict(rep, before=R.J, after=R.J, accepted=0, replays=0, cpu=0.0, secs=0.0, moves=[], stats={})
    if not rep["equal"] or rep["bad_weeks"]:
        out["error"] = "replay mismatch"
        return out
    comp0 = R.components()
    fs = fuel_slots(R.inst)
    v = np.array([R.inst.commodities[k].v for _, k, _ in R.inst.action_slots])
    stats, moves = {}, []
    t1, c1 = time.time(), time.process_time()
    calls0 = R.calls
    out_of = lambda: R.calls - calls0 >= budget or time.time() - t1 > secs  # noqa: E731
    nxt = [500]

    def progress():
        if R.calls - calls0 >= nxt[0]:
            nxt[0] += 500
            logline(f"[ep {n}] replays {R.calls - calls0} accepted {len(moves)} J {R.J / 1e11:.3f} bn "
                    f"(dJ {(out['before'] - R.J) / 1e11:.3f}) {time.time() - t1:.0f}s")
    sweep = 0
    for sweep in range(sweeps):
        improved = False
        cand = sorted((-q * v[s], t, s) for t in range(1, R.T + 1) for s, q in R.flows[t - 1].items() if s in fs and q > 1e-9)
        for _, t, s in cand:
            if out_of():
                break
            for _ in range(repeat):
                hit = False
                for name, ch in moves_of(R, t, s):
                    if out_of():
                        break
                    old = {key: R.flows[key[0] - 1].get(key[1], 0.0) for key in ch}
                    J, flows = R.trial(ch)
                    progress()
                    st = stats.setdefault(name, [0, 0, 0])
                    st[0] += 1
                    if J < R.J:
                        gain = R.J - J
                        R.accept(ch, flows, J)
                        st[1] += 1
                        st[2] += gain
                        moves.append(dict(sweep=sweep, week=t, slot=s, name=slot_name(R.inst, s), fuel=fs[s][0],
                                          type=fs[s][1], move=name, gain=gain,
                                          changes=[(key[0], old[key], ch[key]) for key in sorted(ch)]))
                        improved = hit = True
                        break
                if not hit:
                    break
        if not improved or out_of():
            break
    comp1 = R.components()
    out.update(after=R.J, accepted=len(moves), replays=R.calls - calls0, weeks=R.weeks, secs=time.time() - t1,
               cpu=time.process_time() - c1, total_secs=time.time() - t0, total_cpu=time.process_time() - c0,
               moves=moves, stats=stats, sweeps=sweep + 1, comp0=comp0, comp1=comp1,
               n_fuel_disp=sum(1 for t in range(R.T) for s, q in R.flows[t].items() if s in fs and q > 1e-9))
    with open(done + ".tmp", "wb") as f:
        pickle.dump(dict(out, flows=R.flows), f)
    os.replace(done + ".tmp", done)
    logline(f"[ep {n}] DONE J {out['before'] / 1e11:.3f} -> {out['after'] / 1e11:.3f} bn (dJ {(out['before'] - out['after']) / 1e11:.3f}), "
            f"{out['accepted']} moves, {out['replays']} replays, {out['cpu']:.0f} cpu s")
    return out


# ----- reporting ------------------------------------------------------------------------------------------------------
def references(ns, omega_hashes):
    """Team references for root ENTROPY (computed on the team's instance; see planner_LS.references)."""
    import glob
    import json

    base = glob.glob(os.path.join(ROOT, "hub", "refcache", "references", "*", TASK, "*", f"entropy-{ENTROPY}", "fq-1000-*"))
    base = sorted((b for b in base if all(os.path.exists(os.path.join(b, f"{n}.json")) for n in ns)), key=os.path.getmtime, reverse=True)
    if not base:
        return None, None
    rows, same = [], 0
    for n in ns:
        with open(os.path.join(base[0], f"{n}.json")) as f:
            r = json.load(f)
        same += r["omega_hash"] == omega_hashes[n]
        h = r["harm_usd"]
        r["stratum"] = 1 + sum(h > (lo + hi) / 2 for lo, hi in LS.GAPS)  # small cut points; unused for Full
        rows.append(r)
    return rows, same


def report(res, refs=None):
    bn = 1e11  # cents per bn USD
    print(f"\n{'ep':>3} {'J before bn':>12} {'J after bn':>12} {'dJ bn':>8} {'acc':>4} {'replays':>7} {'cpu s':>6} {'rep/s':>6} "
          f"{'lng':>4} {'crude':>5} {'nuc':>4} {'valve':>5} {'order':>5} " + " ".join(f"{b:>6}" for b in bands()), flush=True)
    for r in res:
        if r.get("error"):
            print(f"{r['n']:>3} ERROR {r['error']}: gym {r['gym_J']} replay {r['replay_J']} bad weeks {r['bad_weeks']}")
            continue
        m = r["moves"]
        c = collections.Counter(x["fuel"] for x in m)
        ty = collections.Counter(x["type"] for x in m)
        b = collections.Counter(band(x["week"]) for x in m)
        print(f"{r['n']:>3} {r['before'] / bn:12.3f} {r['after'] / bn:12.3f} {(r['before'] - r['after']) / bn:8.3f} {r['accepted']:>4} "
              f"{r['replays']:>7} {r['cpu']:>6.0f} {r['replays'] / max(r['cpu'], 1e-9):>6.1f} {c['lng']:>4} {c['crude']:>5} "
              f"{c['nucfuel']:>4} {ty['valve']:>5} {ty['order']:>5} " + " ".join(f"{b[k]:>6}" for k in bands()))
    ok = [r for r in res if not r.get("error")]
    if not ok:
        return
    gains = np.array([(r["before"] - r["after"]) / bn for r in ok])
    print(f"mean dJ {gains.mean():.3f} bn USD/episode (median {np.median(gains):.3f}, min {gains.min():.3f}, max {gains.max():.3f})")
    allm = [x for r in ok for x in r["moves"]]
    for key in ("fuel", "type", "move"):
        g = collections.defaultdict(lambda: [0, 0])
        for x in allm:
            g[x[key]][0] += 1
            g[x[key]][1] += x["gain"]
        print(f"by {key}: " + ", ".join(f"{k} {v[0]} moves {v[1] / bn:.2f} bn" for k, v in sorted(g.items(), key=lambda kv: -kv[1][1])))
    g = collections.defaultdict(lambda: [0, 0])
    for x in allm:
        g[band(x["week"])][0] += 1
        g[band(x["week"])][1] += x["gain"]
    print("by week band: " + ", ".join(f"{k} {v[0]} moves {v[1] / bn:.2f} bn" for k, v in sorted(g.items())))
    g = collections.defaultdict(lambda: [0, 0])
    for x in allm:
        g[(x["type"], x["fuel"], x["name"])][0] += 1
        g[(x["type"], x["fuel"], x["name"])][1] += x["gain"]
    print("top slots by gain:")
    for k, vv in sorted(g.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {k[0]:5s} {k[1]:7s} {k[2]:60s} {vv[0]:4d} moves {vv[1] / bn:7.2f} bn")
    keys = list(ok[0]["comp0"])
    print("cost components, mean change after - before (bn USD/episode): " + ", ".join(
        f"{k} {np.mean([r['comp1'][k] - r['comp0'][k] for r in ok]) / 1e9:+.2f}" for k in keys))
    tried = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        for k, s in r["stats"].items():
            tried[k][0] += s[0]
            tried[k][1] += s[1]
    print("moves accepted/tried: " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in tried.items()))
    if refs is not None and refs[0] is not None:
        rows, same = refs
        idx = {r["episode"]: i for i, r in enumerate(rows)}
        sub = [rows[idx[r["n"]]] for r in ok]
        if TASK == "small":
            b0, _ = LS.rss(sub, [r["before"] for r in ok])
            b1, _ = LS.rss(sub, [r["after"] for r in ok])
            print(f"RSS (SECONDARY; team references, omega hash equal in {same}/{len(rows)} episodes): before {b0:.4f} after {b1:.4f}"
                  f" (+{b1 - b0:.4f})")
        print(f"references (omega hash equal in {same}/{len(rows)}): gap naive-oracle mean {np.mean([(r['J_naive_cents'] - r['J_oracle_cents']) / bn for r in sub]):.1f} bn")
        print("episode gain / (naive-oracle gap), mean over episodes (an RSS-like number, ratio of per-episode values): "
              f"{np.mean([(r['before'] - r['after']) / (x['J_naive_cents'] - x['J_oracle_cents']) for r, x in zip(ok, sub)]):.4f}")


def examples(res, k=3, path=None):
    path = path or os.path.join(HERE, "LSF_examples.md" if TASK == "small" else f"LSF_{TASK}_examples.md")
    bn = 1e11
    ok = sorted((r for r in res if not r.get("error")), key=lambda r: r["after"] - r["before"])[:k]
    lines = ["# LSF: прийняті ходи паливного локального пошуку (planner_LSF.py)", "",
             f"Агент `anastasiia_hybrid_chiplp`, task {TASK}, root {ENTROPY}; {k} епізоди(ів) з найбільшим виграшем. "
             "Ходи в порядку прийняття; «тиждень» — тиждень відправки; кількість — у одиницях товару. "
             "Тип: valve — термінал → система (лаг 0), order — замовлення з джерела. Перевірено одним прогоном.", ""]
    for r in ok:
        lines += [f"## Епізод {r['n']}: J {r['before'] / bn:.2f} → {r['after'] / bn:.2f} млрд USD "
                  f"(−{(r['before'] - r['after']) / bn:.2f}), прийнято {r['accepted']} ходів", "",
                  "| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |", "|---|---|---|---|---|---|---|"]
        for i, x in enumerate(r["moves"], 1):
            ch = "; ".join(f"т{t}: {a:,.0f} → {b:,.0f}" for t, a, b in x["changes"])
            lines.append(f"| {i} | {x['move']} | {x['type']} | {x['fuel']} | `{x['name']}` | {ch} | {x['gain'] / bn:.3f} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def run(episodes=12, start=0, budget=20000, secs=1200, n_jobs=2, probe=2, min_gain_bn=0.5, task="small", entropy=444,
        log=None, nexamples=3):
    """Episodes start..start+episodes-1; the first ``probe`` first, stopping if they fail or gain < min_gain_bn each."""
    configure(task, entropy, log)
    kw = dict(task=task, entropy=entropy, log=log)
    t0 = time.time()
    ns = list(range(start, start + episodes))
    print(f"planner_LSF: {TASK} root {ENTROPY} episodes {ns[0]}..{ns[-1]}, budget {budget} replays / {secs}s, n_jobs {n_jobs}", flush=True)
    res = Parallel(n_jobs=n_jobs)(delayed(search_one)(n, budget, secs, **kw) for n in ns[:probe])
    report(res)
    print(f"[probe done {time.time() - t0:.0f}s]", flush=True)
    if any(r.get("error") for r in res):
        print("STOP: replay mismatch", flush=True)
        return
    if all((r["before"] - r["after"]) / 1e11 < min_gain_bn for r in res):
        print("STOP: near-zero gains on the probe episodes", flush=True)
        return
    res += Parallel(n_jobs=n_jobs)(delayed(search_one)(n, budget, secs, **kw) for n in ns[probe:])
    refs = references(ns, {r["n"]: r["omega_hash"] for r in res})
    print(f"\n===== all {len(res)} episodes ({time.time() - t0:.0f}s wall) =====")
    report(res, refs)
    examples(res, k=nexamples)
    with open(os.path.join(CACHE, "lsf_all.pkl"), "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    fire.Fire({"run": run, "verify": verify})
