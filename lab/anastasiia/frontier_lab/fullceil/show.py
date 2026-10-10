"""The Full ladder with the whole future, by episode and in all (notes/u_fullceil.md).

    uv run python lab/anastasiia/frontier_lab/fullceil/show.py
    uv run python lab/anastasiia/frontier_lab/fullceil/show.py --only=2,3,9 --stages

Columns are bn USD an episode. "played": the simulator's cost of executed actions. "own value": what the integer
program says its plan costs (a relaxation: the simulator's other automatic rules are free in it). "bound": the integer
program's dual bound (no agent can be cheaper). The integer program's numbers are the better of the run kept here
(``milp.py``, started from an executed plan) and the 300 s run of 6 October
(``outputs/heur4/lead/plan_full_444_x20/episodes.npz``, ``stats_lab/plan_stats.py``; no episode 3).
"""

import pickle

import common as K
import fire
import numpy as np
from record import start_path


def old_plans(task: str, entropy: int, refs: list[dict]) -> dict:
    """Episode -> (the kept 300 s integer plan's own value, its dual bound), cents; aligned by the clairvoyant cost."""
    path = K.ROOT / "outputs" / "heur4" / "lead" / "plan_full_444_x20" / "episodes.npz"
    if task != "full" or entropy != 444 or not path.is_file():
        return {}
    z = np.load(path)
    out = {}
    for i, jr in enumerate(z["J_relaxed"] * 100.0):
        for n in range(20):
            if abs(refs[n]["J_oracle_cents"] - jr) < 1e-6 * jr:
                out[n] = (float(z["J_plan"][i]) * 100.0, float(z["J_plan"][i] * (1.0 - z["mip_gap"][i])) * 100.0)
    return out


def collect(task: str, entropy: int, ns: list[int]) -> dict:
    refs = K.references(task, entropy, 64)
    kept = {tag: K.kept(tag, task, entropy) for tag in ("h3c_f", "h3_f", "truthallc_f", "tah0c_f")}
    old = old_plans(task, entropy, refs)
    rows = {}
    for n in ns:
        row = {"level": refs[n]["stratum"], "naive": refs[n]["J_naive_cents"], "clair": refs[n]["J_oracle_cents"]}
        for tag, d in kept.items():
            if n in d:
                row[tag] = d[n]["J"]
        for tag in ("tah0_f", "h3_f"):
            p = start_path(tag, task, entropy, n)
            if p.is_file():
                row[f"rec:{tag}"] = pickle.loads(p.read_bytes())["J"]
        row["runs"] = {}
        for f in sorted((K.OUT / "runs").glob(f"*_{task}_{entropy}_{n}.pkl")):
            d = pickle.loads(f.read_bytes())
            if "J" in d:
                row["runs"][f.name[: -len(f"_{task}_{entropy}_{n}.pkl")]] = d
        row["milps"] = {}
        for f in sorted((K.OUT / "milp").glob(f"*_{task}_{entropy}_{n}.pkl")):
            name = f.name[: -len(f"_{task}_{entropy}_{n}.pkl")]
            if name == "fixed":  # the program at the executed plan's own whole weeks (``fixed.py``)
                row["fixed"] = pickle.loads(f.read_bytes())
            else:
                row["milps"][name] = pickle.loads(f.read_bytes())
        row["old"] = old.get(n)
        rows[n] = row
    return rows


def best_executed(row: dict) -> tuple[float, str]:
    """The cheapest played cost known for the episode with the whole future, cents, and where it came from."""
    cands = {f"run {k}": d["J"] for k, d in row["runs"].items()}
    for k in ("tah0c_f", "rec:tah0_f"):
        if k in row:
            cands[k] = row[k]
    for k, d in row["milps"].items():
        cands[f"{k} played blind"] = d["J_played"]
    k = min(cands, key=cands.get)
    return float(cands[k]), k


def own_value(row: dict) -> float:
    """The integer program's cheapest own value known, cents (never dearer than an executed plan: each is feasible)."""
    cands = [best_executed(row)[0]] + [d["J"] * 100.0 for d in row["milps"].values()] + ([row["old"][0]] if row["old"] else [])
    return float(min(cands))


def bound(row: dict) -> float:
    """The largest dual bound known, cents; nan without one."""
    cands = [d["bound"] * 100.0 for d in row["milps"].values()] + ([row["old"][1]] if row["old"] else [])
    return float(max(cands)) if cands else np.nan


def main(only: str | tuple | int = "", task: str = "full", entropy: int = 444, first: int = 0, episodes: int = 16,
         draws: int = 4000, stages: bool = False, need: str = "") -> None:
    """``need``: only the episodes that have a finished descent under this label enter the sums (default: any run)."""
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else list(range(first, first + episodes))
    rows = collect(task, entropy, ns)
    print(f"{task} root {entropy}; bn USD an episode; played = the simulator's cost; score = (naive - cost) / room of the episode")
    print("ep lvl   room | h3c_f  told-window  told-all kept / here | best executed (from; to told-all kept, to told-all here) | "
          "MILP own value, bound | scores: told-all kept, best executed, own value, bound")
    done = []
    for n in ns:
        r = rows[n]
        room = (r["naive"] - r["clair"]) / K.BN
        best, who = best_executed(r)
        rec = r.get("rec:tah0_f")

        def sc(J: float) -> str:
            return f"{(r['naive'] - J) / (r['naive'] - r['clair']):.4f}" if np.isfinite(J) else "   -  "

        print(f"{n:2d}  {r['level']}  {room:6.0f} | {r['h3c_f'] / K.BN:8.1f} {r['truthallc_f'] / K.BN:8.1f} {r['tah0c_f'] / K.BN:8.1f} / "
              f"{(f'{rec / K.BN:8.1f}' if rec else '       -')} | {best / K.BN:8.1f} ({who}; {(r['tah0c_f'] - best) / K.BN:+6.1f}, "
              f"{(f'{(rec - best) / K.BN:+6.1f}' if rec else '     -')}) | {own_value(r) / K.BN:8.1f} {bound(r) / K.BN:8.1f} | "
              f"{sc(r['tah0c_f'])} {sc(best)} {sc(own_value(r))} {sc(bound(r))}")
        if stages:
            for k, d in r["runs"].items():
                print(f"      {k}{'' if d.get('done') else ' (running)'}: start {d['J0'] / K.BN:.1f}" + "".join(
                    f" -> {s['name']} {s['J'] / K.BN:.1f}" + (f" [{s['search'][1]}]" if s.get("search") else "") for s in d["stages"] if s["name"] != "end")
                    + f"  ({d.get('seconds', 0):.0f} s CPU)")
            for k, d in r["milps"].items():
                print(f"      {k}: from '{d['start']}' {d['J_exec'] / K.BN:.1f} played; own value {d['J'] / 1e9:.1f}, bound {d['bound'] / 1e9:.1f} "
                      f"(gap {100 * d['gap']:.2f} %, {d['seconds']:.0f} s, {d['nodes']} nodes); its actions played blind {d['J_played'] / K.BN:.1f}")
            if "fixed" in r:
                print(f"      the program at the executed plan's whole weeks: {r['fixed']['J'] / 1e9:.1f} "
                      f"({r['fixed']['J_exec'] / K.BN - r['fixed']['J'] / 1e9:.1f} under the played {r['fixed']['J_exec'] / K.BN:.1f})")
            if r["old"]:
                print(f"      300 s run of 6 October: own value {r['old'][0] / K.BN:.1f}, bound {r['old'][1] / K.BN:.1f}")
        if (need in r["runs"] and r["runs"][need].get("done")) if need else bool(r["runs"]):
            done.append(n)
    if not done:
        return
    main_runs = [n for n in done if "tah0_f" in rows[n]["runs"] and rows[n]["runs"]["tah0_f"].get("done")]
    if main_runs:  # the descent from the told planner's own play, by stage: what each stage takes off, bn
        print("\nthe descent from the told planner's play recorded here, bn taken off by stage (cells to a fixed point; rounds of the hull with the search)")
        tot = np.zeros(3)
        for n in main_runs:
            d = rows[n]["runs"]["tah0_f"]
            st = {s["name"]: s["J"] for s in d["stages"]}
            plain = (d["J0"] - st["descent"]) / K.BN
            hullg = (st["descent"] - d["J"]) / K.BN
            moves = "; ".join(f"{s['name']}: {s['search'][1]}" for s in d["stages"] if s.get("search") and s["name"] != "end")
            tot += [plain, hullg, (rows[n]["tah0c_f"] - d["J0"]) / K.BN]
            print(f"  ep {n:2d}: kept play minus the play here {(rows[n]['tah0c_f'] - d['J0']) / K.BN:+7.1f} | cells {plain:6.1f} | hull and search {hullg:6.1f} | "
                  f"{d.get('seconds', 0):5.0f} s | {moves}")
        print(f"  mean of {len(main_runs)}: kept minus here {tot[2] / len(main_runs):+.1f} | cells {tot[0] / len(main_runs):.1f} | hull and search {tot[1] / len(main_runs):.1f}")
    level = np.array([rows[n]["level"] for n in done])
    naive = np.array([rows[n]["naive"] for n in done], dtype=float)
    clair = np.array([rows[n]["clair"] for n in done], dtype=float)

    def col(f) -> np.ndarray:
        return np.array([f(rows[n]) for n in done], dtype=float)

    ladder = [("model h3c_f (played)", col(lambda r: r["h3c_f"])),
              ("told the window, truthallc_f (played)", col(lambda r: r["truthallc_f"])),
              ("told everything, tah0c_f (played)", col(lambda r: r["tah0c_f"]))]
    if all("rec:tah0_f" in rows[n] for n in done):
        ladder.append(("told everything, played again here (not a step)", col(lambda r: r["rec:tah0_f"])))
    ladder.append(("best executed with the whole future (played)", col(lambda r: best_executed(r)[0])))
    if all(np.isfinite(bound(rows[n])) for n in done):
        ladder += [("integer program, own value (not played)", col(own_value)), ("integer program, dual bound", col(bound))]
    ladder.append(("clairvoyant", clair))
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(level == s) for s in sorted(set(level.tolist()))]
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    weighted = set(level.tolist()) == set(K.WEIGHTS)
    print(f"\n{len(done)} episodes {done}, by harm level {[int((level == s).sum()) for s in K.WEIGHTS]}; score: "
          + ("the board's level weights" if weighted else "all saved over all room (a level is missing)") + "; and the plain ratio of sums")
    prev = None
    for name, J in ladder:
        s = K.rss(level, naive, clair, J)
        plain = float((naive - J).sum() / (naive - clair).sum())
        line = f"  {name:48s} {s:.4f}  plain {plain:.4f}  mean cost {J.mean() / K.BN:9.1f} bn"
        if prev is not None and "not a step" not in name:
            v = [K.rss(level[p], naive[p], clair[p], J[p]) - K.rss(level[p], naive[p], clair[p], prev[p]) for p in picks]
            lo, hi = np.percentile(v, [5, 95])
            line += f"  step {s - K.rss(level, naive, clair, prev):+.4f} ({lo:+.4f} to {hi:+.4f}), {(prev - J).mean() / K.BN:+7.1f} bn an episode"
        print(line)
        if "not a step" not in name:
            prev = J


if __name__ == "__main__":
    fire.Fire(main)
