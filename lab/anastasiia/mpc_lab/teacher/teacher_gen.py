"""Privileged teacher for imitation learning: (gym observation, teacher action) pairs on training episodes.

    uv run python lab/anastasiia/mpc_lab/teacher/teacher_gen.py generate --task=small --entropy=555 --episodes=32 --start=0 \
        --tl=60 --ls_budget=5000 --n_jobs=2
    uv run python lab/anastasiia/mpc_lab/teacher/teacher_gen.py report --task=small --entropy=555

Per episode n of (task, entropy) the teacher knows the episode's future (legal: we generate the training episodes):

1. MILP. planner_H's full-episode plan with the exact simulator rules (base, fuel, lots), lazy constraint generation
   (``planner_LS.solve_lazy_keep``: planner_H.solve_lazy, but a round without an incumbent keeps the previous round's
   plan). If even round 0 has no solution, the fallback is the plan's LP without the rules (no binaries).
2. Played plan. The plan's weekly wire actions (``lp_common.week1_action`` per week, ``planner_LS.plan_actions``) in the
   core Env: the played J before the search. The wire actions are cached in <out>/stage/ep<n>_plan.pkl (a restart does
   not solve the MILP again).
3. Local search. planner_LS's moves (x1.5, drop, shift +-1/+-2, half, x2 ...) on the flows of every action slot (wafer
   slots first, then fuel, then the rest, each by qty x customs value), strict first improvement, judged by the fast
   replay (raw ``dynamics.sim.step`` from per-week checkpoints). Overrides / holds at chokepoints stay as the plan's.
4. Recording. The final actions are played through the GYMNASIUM env (``gym.make(env_id(task), entropy=E)``,
   ``reset(options={"episode": n})``, policy seed 0, no fallback) and each week's flat observation (the dict
   ``env.step`` returns, as-is) is stored with the week's action in gym format. The gym J must equal the replay's J.
5. References. naive's and the clairvoyant plan's costs from ``sbf_starter.scoring.episode_set(task, [n], entropy=E)``
   (``sbf_starter`` imported first, so hub/refcache is used and extended), for the teacher's RSS.

Output: outputs/teacher/<task>_<entropy>/ep<n>.npz (format in README.md), config.pkl.gz (the agent config, identical
for every episode of a task). Progress: lab/anastasiia/mpc_lab/teacher/gen.log (appended per step).
"""

import gzip
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sbf_starter  # noqa: E402,F401  (first: sets SBF_CACHE_DIR to hub/refcache)

import fire  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "planners"), os.path.dirname(HERE)]
import planner_H as H  # noqa: E402
import planner_LS as LS  # noqa: E402
from package_baselines import rss  # noqa: E402

LOG = os.path.join(HERE, "gen.log")
RESERVED = {0, 111, 222, 333, 444}
FORMAT_VERSION = 1


def configure(task, entropy):
    """Point planner_H / planner_LS at (task, entropy); called in every joblib worker (modules import afresh there)."""
    H.TASK, H.ENTROPY = task, int(entropy)
    LS.patch()  # H.world = planner_LS.world_nofq: (instance, omega of (H.TASK, H.ENTROPY, n), marks, no fallback)


def out_dir(task, entropy):
    # TEACHER_MEASURE reaches the joblib workers through the environment (a global would not survive the spawn)
    prefix = "measure_" if os.environ.get("TEACHER_MEASURE") else ""
    return os.path.join(ROOT, "outputs", "teacher", f"{prefix}{task}_{entropy}")


def logline(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid {os.getpid()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def atomic_pickle(obj, path):
    with gzip.open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)


# ----- 1-2. MILP plan and its played actions ---------------------------------------------------------------------------
def plan_stage(n, task, entropy, rules, tl, gap, rounds):
    """The plan's wire actions and its numbers, cached in <out>/stage/ep<n>_plan.pkl.gz."""
    stage = os.path.join(out_dir(task, entropy), "stage")
    path = os.path.join(stage, f"ep{n}_plan.pkl.gz")
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        logline(f"[{task} {entropy} ep {n}] plan loaded from stage: claimed {d['claimed']} played {d['played']} ({d['status']})")
        return d
    t0, c0 = time.time(), time.process_time()
    best, hist, status = lazy_best(n, H.parse(rules), tl, gap, rounds, f"[{task} {entropy} ep {n}]")
    if best is None:
        logline(f"[{task} {entropy} ep {n}] MILP: no solution in round 0 (tl {tl}s); fallback: the plan's LP without rules")
        out, sol = H.solve_full(n, (), tl=max(tl, 300), gap=gap, world_=H.world(n))
        if sol is None:
            raise RuntimeError(f"episode {n}: neither the MILP nor the LP has a solution")
        acts, J, _R = LS.plan_actions(n, *sol)
        best, status = dict(round=-1, claimed=int(out["J"]), played=int(J), actions=acts, gap=out.get("gap")), "lp_fallback"
    d = dict(n=n, claimed=best["claimed"], played=best["played"], actions=best["actions"], gap=best["gap"],
             rounds=len(hist), chosen_round=best["round"], hist=hist, status=status, milp_secs=time.time() - t0,
             milp_cpu=time.process_time() - c0)
    os.makedirs(stage, exist_ok=True)
    atomic_pickle(d, path)
    logline(f"[{task} {entropy} ep {n}] MILP {status}: chose round {d['chosen_round']} of {d['rounds']}: claimed J "
            f"{d['claimed'] / 1e11:.3f} bn, played J {d['played'] / 1e11:.3f} bn, gap {d['gap']}, {d['milp_secs']:.0f}s")
    return d


def lazy_best(n, rules, tl, gap, rounds, tag):
    """planner_H.solve_lazy's constraint generation, each round's plan played in the Env; the best PLAYED plan is kept.

    A round without an incumbent in ``tl`` ends the loop (as planner_LS.solve_lazy_keep). Returns (best, hist, status):
    best = dict(round, claimed, played, actions, gap) or None when round 0 already failed.
    """
    from shockbench_flow.dynamics.sim import initial_stock

    w = H.world(n)
    inst = w[0]
    i0 = initial_stock(inst)
    gw, ow, pref, best, hist, status = set(), set(), None, None, [], "round_cap"
    for it in range(rounds + 1):
        out, sol = H.solve_full(n, rules, tl, gap, only_gw=gw, only_ow=ow, world_=w, phat_ref=pref)
        if sol is None:
            logline(f"{tag} round {it}: no incumbent in {tl}s; stop")
            status = f"no_incumbent_round_{it}"
            break
        m, x = sol
        acts, J, _R = LS.plan_actions(n, m, x)
        g2, o2, kinds, pref = H.violations(inst, m, x, i0)
        hist.append(dict(round=it, gw=len(gw), ow=len(ow), new_gw=len(g2 - gw), new_ow=len(o2 - ow), secs=out["secs"],
                         gap=out["gap"], claimed=int(out["J"]), played=int(J)))
        logline(f"{tag} round {it}: binaries on {len(gw)}/{len(ow)} grid/osat-weeks, new violations {len(g2 - gw)}/{len(o2 - ow)}, "
                f"gap {out['gap']}, claimed {out['J'] / 1e11:.3f} played {J / 1e11:.3f} bn, {out['secs']:.0f}s")
        if best is None or J < best["played"]:
            best = dict(round=it, claimed=int(out["J"]), played=int(J), actions=acts, gap=out["gap"])
        if not (g2 - gw) and not (o2 - ow):
            status = "converged"
            break
        gw |= g2
        ow |= o2
    return best, hist, status


# ----- 3. local search on the played actions ---------------------------------------------------------------------------
def local_search(R, budget, secs, order="kind", sweeps=3, repeat=3, tag=""):
    """planner_LS.search_one's loop on a Replay ``R`` (modified in place). Returns stats."""
    t0, c0 = time.time(), time.process_time()
    J0, calls0 = R.J, R.calls
    kinds = LS.slot_kinds(R.inst)
    stats, bykind, acc = {}, {}, 0
    out_of = lambda: R.calls - calls0 >= budget or time.time() - t0 > secs  # noqa: E731
    nxt = [1000]
    sweep = 0
    for sweep in range(sweeps):
        improved = False
        for t, s in LS.candidates(R, kinds, order):
            if out_of():
                break
            for _ in range(repeat):
                hit = False
                for name, ch in LS.moves_of(R, t, s):
                    if out_of():
                        break
                    J, flows = R.trial(ch)
                    if R.calls - calls0 >= nxt[0]:
                        nxt[0] += 1000
                        logline(f"{tag} LS replays {R.calls - calls0} accepted {acc} J {R.J / 1e11:.3f} bn "
                                f"(dJ {(J0 - R.J) / 1e11:.3f}) {time.time() - t0:.0f}s")
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
                        break
                if not hit:
                    break
        if not improved or out_of():
            break
    return dict(before=J0, after=R.J, accepted=acc, replays=R.calls - calls0, sweeps=sweep + 1, secs=time.time() - t0,
                cpu=time.process_time() - c0, stats=stats, bykind=bykind)


# ----- 4. recording through gymnasium ---------------------------------------------------------------------------------
def gym_actions(layout, wires, flows):
    """Gym-format actions (flows, override_qty, release_mode) per week: the searched flows, the plan's overrides/holds."""
    from shockbench_flow.information.flat import flat_from_action

    F, OQ, RM = [], [], []
    for w, fl in zip(wires, flows):
        _f, oq, rm = flat_from_action(layout, w)
        f = np.zeros(layout.n_slots, dtype=np.float64)
        for s, q in fl.items():
            f[s] = q
        F.append(f), OQ.append(oq), RM.append(rm)
    return np.stack(F), np.stack(OQ), np.stack(RM)


def record(task, entropy, n, wires, flows, cfg_path):
    """Play the final actions in the gym env; per-week observations, actions, weekly reward cents, J."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    if not os.path.exists(cfg_path):
        atomic_pickle(agent_config(info["static"], info["policy_seed"], u.layout, obs), cfg_path)
    F, OQ, RM = gym_actions(u.layout, wires, flows)
    obs_list, rewards, done, t = [], [], False, 0
    while not done:
        obs_list.append(obs)
        obs, _r, term, trunc, inf = env.step({"flows": F[t], "override_qty": OQ[t], "release_mode": RM[t]})
        rewards.append(int(inf["reward_cents"]))
        done = term or trunc
        t += 1
    traj = u.core.trajectory
    invalid = sum(len(r.invalid) for r in traj.records)
    stacked = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    return dict(obs=stacked, flows=F, override_qty=OQ, release_mode=RM, reward_cents=np.array(rewards, dtype=np.int64),
                J=int(traj.J_cents), salvage_cents=int(traj.salvage_cents), omega_hash=traj.omega_hash, invalid=invalid)


# ----- one episode ----------------------------------------------------------------------------------------------------
def episode(n, task, entropy, rules, tl, gap, rounds, ls_budget, ls_secs, order):
    configure(task, entropy)
    from shockbench_flow.dynamics.env import validate_action  # noqa: F401
    from planner_LSF import FReplay
    from sbf_starter import scoring

    d_out = out_dir(task, entropy)
    path = os.path.join(d_out, f"ep{n}.npz")
    if os.path.exists(path):
        return summary_of(path)
    tag = f"[{task} {entropy} ep {n}]"
    logline(f"{tag} start")
    t0, c0 = time.time(), time.process_time()
    plan = plan_stage(n, task, entropy, rules, tl, gap, rounds)
    inst, omega, marks, _ = H.world(n)
    R = FReplay(inst.at_digest(marks.instance_digest), marks, plan["actions"])
    if R.J != plan["played"]:
        logline(f"{tag} WARNING replay J {R.J} != Env played J {plan['played']} (dropped entries {R.dropped})")
    ls = local_search(R, ls_budget, ls_secs, order=order, tag=tag)
    logline(f"{tag} LS done: J {ls['before'] / 1e11:.3f} -> {ls['after'] / 1e11:.3f} bn (dJ {(ls['before'] - ls['after']) / 1e11:.3f}), "
            f"{ls['accepted']} moves, {ls['replays']} replays, {ls['cpu']:.0f} cpu s")
    rec = record(task, entropy, n, plan["actions"], R.flows, os.path.join(d_out, "config.pkl.gz"))
    if rec["J"] != R.J:
        logline(f"{tag} WARNING gym J {rec['J']} != replay J {R.J} (invalid entries {rec['invalid']})")
    ref = scoring.episode_set(task, [n], entropy=entropy, n_jobs=1, verbose=False).references[0]
    if ref["omega_hash"] != rec["omega_hash"]:
        logline(f"{tag} WARNING omega hash of the references differs from the gym episode's")
    meta = dict(format=FORMAT_VERSION, task=task, entropy=entropy, episode=n, rules=rules, tl=tl, gap=gap, rounds=rounds,
                ls_budget=ls_budget, order=order, milp_status=plan["status"], milp_gap=plan["gap"], milp_rounds=plan["rounds"], milp_chosen_round=plan["chosen_round"],
                milp_hist=plan["hist"], milp_secs=plan["milp_secs"], ls_stats=ls["stats"], ls_bykind=ls["bykind"],
                ls_accepted=ls["accepted"], ls_replays=ls["replays"], ls_sweeps=ls["sweeps"], ls_secs=ls["secs"],
                total_secs=time.time() - t0, total_cpu=time.process_time() - c0, gym_invalid_entries=rec["invalid"],
                ref_harm_usd=ref["harm_usd"], ref_excluded=ref["excluded"], omega_hash=rec["omega_hash"],
                ref_omega_hash=ref["omega_hash"])
    arrays = {f"obs/{k}": v for k, v in rec["obs"].items()}
    arrays.update({
        "act/flows": rec["flows"], "act/override_qty": rec["override_qty"], "act/release_mode": rec["release_mode"],
        "reward_cents": rec["reward_cents"],
        "J_teacher": np.int64(rec["J"]), "J_replay": np.int64(R.J), "J_plan_played": np.int64(plan["played"]),
        "J_claimed": np.int64(plan["claimed"]), "J_naive": np.int64(ref["J_naive_cents"]),
        "J_oracle": np.int64(ref["J_oracle_cents"]), "stratum": np.int64(ref["stratum"] or 0),
        "salvage_cents": np.int64(rec["salvage_cents"]), "meta": np.array(json.dumps(meta, default=str)),
    })
    tmp = os.path.join(d_out, f"ep{n}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    s = summary_of(path)
    logline(f"{tag} SAVED {os.path.getsize(path) / 1e6:.1f} MB: claimed {s['claimed'] / 1e11:.3f} played {s['played'] / 1e11:.3f} "
            f"teacher {s['J'] / 1e11:.3f} bn | RSS-episode claimed {s['rss_claimed']:.4f} played {s['rss_played']:.4f} "
            f"teacher {s['rss_teacher']:.4f} | {meta['total_secs']:.0f}s")
    return s


def ep_rss(Jn, Jo, J):
    return (Jn - J) / (Jn - Jo)


def summary_of(path):
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        Jn, Jo = int(z["J_naive"]), int(z["J_oracle"])
        s = dict(n=meta["episode"], J=int(z["J_teacher"]), replay=int(z["J_replay"]), played=int(z["J_plan_played"]),
                 claimed=int(z["J_claimed"]), J_naive_cents=Jn, J_oracle_cents=Jo, stratum=int(z["stratum"]),
                 status=meta["milp_status"], gap=meta["milp_gap"], secs=meta["total_secs"], milp_secs=meta["milp_secs"],
                 ls_secs=meta["ls_secs"], accepted=meta["ls_accepted"], replays=meta["ls_replays"],
                 size=os.path.getsize(path), T=int(z["act/flows"].shape[0]))
    s.update(rss_claimed=ep_rss(Jn, Jo, s["claimed"]), rss_played=ep_rss(Jn, Jo, s["played"]), rss_teacher=ep_rss(Jn, Jo, s["J"]))
    return s


# ----- CLI ------------------------------------------------------------------------------------------------------------
def generate(task="small", entropy=555, episodes=32, start=0, tl=60, gap=1e-3, rounds=8, ls_budget=5000, ls_secs=3600,
             order="kind", rules="base,fuel,lots", n_jobs=2, measure=False):
    """Episodes start..start+entropy-1 of (task, entropy); an episode whose ep<n>.npz exists is skipped.

    ``measure`` allows a reserved root (e.g. 111) for CEILING MEASUREMENT ONLY: the output goes to a separate
    measure_<task>_<entropy>/ folder and must never be used as training data.
    """
    if measure:
        os.environ["TEACHER_MEASURE"] = "1"
    if int(entropy) in RESERVED and not measure:
        raise ValueError(f"root {entropy} is reserved (0/111/222/333/444): training data must not overlap them")
    os.makedirs(out_dir(task, entropy), exist_ok=True)
    if isinstance(rules, (tuple, list)):
        rules = ",".join(rules)
    ns = list(range(start, start + episodes))
    logline(f"generate {task} root {entropy} episodes {ns[0]}..{ns[-1]}: tl {tl} gap {gap} rounds {rounds} rules {rules} "
            f"ls_budget {ls_budget} order {order} n_jobs {n_jobs}")
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(
        delayed(episode)(n, task, entropy, rules, tl, gap, rounds, ls_budget, ls_secs, order) for n in ns)
    logline(f"generate done: {len(res)} episodes in {time.time() - t0:.0f}s wall")
    report(task, entropy)


def report(task="small", entropy=555):
    """Teacher quality over every ep<n>.npz of the dataset."""
    d = out_dir(task, entropy)
    files = sorted((f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz") and ".tmp" not in f),
                   key=lambda f: int(f[2:-4]))
    S = [summary_of(os.path.join(d, f)) for f in files]
    if not S:
        print("no episodes")
        return
    bn = 1e11
    print(f"{'ep':>3} {'lvl':>3} {'status':>20} {'claimed bn':>11} {'played bn':>10} {'teacher bn':>11} {'LS dJ bn':>9} "
          f"{'rss cl':>7} {'rss pl':>7} {'rss T':>7} {'acc':>4} {'secs':>5} {'milp s':>6} {'MB':>5}")
    for s in S:
        print(f"{s['n']:>3} {s['stratum']:>3} {s['status']:>20} {s['claimed'] / bn:11.3f} {s['played'] / bn:10.3f} {s['J'] / bn:11.3f} "
              f"{(s['played'] - s['J']) / bn:9.3f} {s['rss_claimed']:7.4f} {s['rss_played']:7.4f} {s['rss_teacher']:7.4f} "
              f"{s['accepted']:>4} {s['secs']:5.0f} {s['milp_secs']:6.0f} {s['size'] / 1e6:5.1f}")
    rows = [dict(J_naive_cents=s["J_naive_cents"], J_oracle_cents=s["J_oracle_cents"], stratum=s["stratum"]) for s in S]
    for name, key in (("claimed (MILP)", "claimed"), ("played plan", "played"), ("teacher (after LS)", "J")):
        sc, lv = rss(rows, [s[key] for s in S])
        print(f"RSS {name:20s} {sc:.4f}  levels " + " ".join(f"{k}:{v:.3f}" for k, v in lv.items()))
    lift = np.array([(s["played"] - s["J"]) / bn for s in S])
    rl = np.array([s["rss_teacher"] - s["rss_played"] for s in S])
    print(f"LS lift: mean {lift.mean():.3f} bn USD/episode (median {np.median(lift):.3f}, min {lift.min():.3f}, max {lift.max():.3f}); "
          f"per-episode RSS lift mean {rl.mean():+.4f}")
    print(f"levels present {sorted(set(s['stratum'] for s in S))} (counts {[sum(s['stratum'] == k for s in S) for k in (1, 2, 3, 4)]}); "
          f"status {dict((k, sum(s['status'] == k for s in S)) for k in set(s['status'] for s in S))}")
    print(f"time/episode: mean {np.mean([s['secs'] for s in S]):.0f}s (MILP {np.mean([s['milp_secs'] for s in S]):.0f}s, "
          f"LS {np.mean([s['ls_secs'] for s in S]):.0f}s); size/episode mean {np.mean([s['size'] for s in S]) / 1e6:.2f} MB; "
          f"gym J == replay J in {sum(s['J'] == s['replay'] for s in S)}/{len(S)}")


if __name__ == "__main__":
    fire.Fire({"generate": generate, "report": report})
