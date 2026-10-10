"""Offline descents with the whole future on Full (notes/u_fullceil.md): the family of the Small ceiling
(``frontier_lab/opening.py``'s ``_descent``: the loop of exact cells, then rounds of whole weeks asked for by the hull
with the search over sets of whole weeks), from a start named by ``--start``. Every cost is the simulator's.

    uv run python lab/anastasiia/frontier_lab/fullceil/descend.py --start=tah0_f --only=2,3,9
    uv run python lab/anastasiia/frontier_lab/fullceil/descend.py --start=free:tah0c_f --only=2     # a kept play's dispatches, default releases
    uv run python lab/anastasiia/frontier_lab/fullceil/descend.py --start=milp --only=2            # the integer plan's own actions
    uv run python lab/anastasiia/frontier_lab/fullceil/descend.py --start=run:tah0_f --hull=6 --label=tah0_f_more --only=2

Starts: ``<tag>`` - the actions ``record.py`` kept (replayed to the cent first); ``free:<tag>`` - the dispatches of a
play kept by ``hazard_lab/play.py`` with the straits' default releases; ``milp`` - the actions of the integer plan kept
by ``milp.py``; ``run:<label>`` - where the descent kept under ``<label>`` ended (more rounds, or a wider search).
One file a (label, episode) in ``outputs/frontier_lab/fullceil/runs/``, written after every stage; a finished one is not
run again. On Full every cell is solved by the interior point with the crossover (116 thousand columns).
"""

import pickle
import time
from pathlib import Path

import common as K
import fire
from record import start_path


def run_path(label: str, task: str, entropy: int, n: int) -> Path:
    return K.OUT / "runs" / f"{label.replace(':', '-')}_{task}_{entropy}_{n}.pkl"


def milp_path(task: str, entropy: int, n: int, name: str = "milp") -> Path:
    return K.OUT / "milp" / f"{name}_{task}_{entropy}_{n}.pkl"


def start_of(ep, start: str, task: str, entropy: int, n: int) -> tuple[list, int | None]:
    """(weekly actions, the cost the start is known to have played to, or None)."""
    if start.startswith("free:"):
        return K.sent_actions(ep, K.kept(start[5:], task, entropy)[n]["sent"]), None
    if start.startswith("run:"):
        d = pickle.loads(run_path(start[4:], task, entropy, n).read_bytes())
        return d["acts"], d["J"]
    if start == "milp" or start.startswith("milp:"):
        d = pickle.loads(milp_path(task, entropy, n, start.replace(":", "_")).read_bytes())
        return d["acts"], d["J_played"]
    d = pickle.loads(start_path(start, task, entropy, n).read_bytes())
    return ep.validated(d["actions"]), d["J"]


def main(start: str = "tah0_f", label: str = "", task: str = "full", entropy: int = 444, only: str | tuple | int = "",
         first: int = 0, episodes: int = 16, iters: int = 60, hull: int = 3, search: int = 8, tol_retry: float = 0.0,
         min_gain: float = 1e9) -> None:
    """``min_gain``: cents; a descent stops after three passes in a row that gain less. The family's own 1e6 (10
    thousand USD) keeps a Full descent crawling to its 60th pass at 0.001 bn a pass, 13 s each; 1e9 is 0.01 bn."""
    import core  # regime_lab's

    label = label or start
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else range(first, first + episodes)
    for n in ns:
        path = run_path(label, task, entropy, n)
        if path.is_file() and pickle.loads(path.read_bytes()).get("done"):
            continue
        t0, w0 = time.process_time(), time.time()
        ep = core.Episode.of(task, entropy, n)
        if tol_retry:
            ep.tol_retry = tol_retry
        method = "ipm" if ep.N > 4 * core.BIG else "simplex"
        acts, J_known = start_of(ep, start, task, entropy, n)
        recs0, J0 = ep.simulate(acts)
        note = "" if J_known is None or J_known == J0 else f"  (the start was kept as {J_known / K.BN:.2f} bn)"
        print(f"{label} ep {n}: start {J0 / K.BN:.2f} bn{note}", flush=True)
        out = {"start": start, "J0": int(J0), "J_known": J_known, "stages": [], "weeks0": K.weeks_of(recs0), "min_gain": min_gain,
               "sent0": K.sent_of(ep, recs0), "done": False}

        def keep(d: dict, name: str, done: bool = False, extra: dict | None = None) -> None:
            out["stages"].append({"name": name, "J": int(d["J"]), "cpu": time.process_time() - t0, **(extra or {})})
            out.update(J=int(d["J"]), acts=d["acts"], hist=d["hist"], weeks=K.weeks_of(d["recs"]), sent=K.sent_of(ep, d["recs"]),
                       seconds=time.process_time() - t0, wall=time.time() - w0, peak_mb=K.rss_mb(), done=done)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(out))
            last = d["hist"][-1] if d["hist"] else None
            print(f"{label} ep {n}: {name:10s} {d['J'] / K.BN:10.2f} bn  ({len(d['hist'])} passes, {time.process_time() - t0:.0f} s CPU, "
                  f"peak {K.rss_mb():.0f} MB; last pass {last[0] if last and last[1] is None else 'played'}"
                  + (f"; {extra}" if extra else "") + ")", flush=True)

        d = core.descend(ep, acts, iters, min_gain=min_gain, played=(recs0, J0), method=method)
        keep(d, "descent")
        for i in range(hull):
            d2 = core.descend(ep, d["acts"], iters=iters, min_gain=min_gain, hull="round", close_until=ep.T - 12, method=method,
                              search=search)
            extra = {"hull": d2.get("hull"), "rounded": d2.get("rounded"), "search": d2.get("search"), "fracy": d2.get("fracy")}
            if d2["J"] > d["J"] - 1e8:
                out["stages"].append({"name": f"hull {i + 1} (no gain)", "J": int(d2["J"]), "cpu": time.process_time() - t0, **extra})
                break
            d = {**d2, "J0": d["J0"], "hist": d["hist"] + d2["hist"]}
            keep(d, f"hull {i + 1}", extra=extra)
        keep(d, "end", done=True)


if __name__ == "__main__":
    fire.Fire(main)
