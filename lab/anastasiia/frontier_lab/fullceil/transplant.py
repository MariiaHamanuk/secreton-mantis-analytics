"""Is the integer plan's set of whole weeks executable on Full? Its whole weeks are asked for in the team's own cell
from the best executed plan (``frontier_lab/audit/transplant.py`` on Small), then the descent runs from what the cell
gives. Every cost is the simulator's.

    uv run python lab/anastasiia/frontier_lab/fullceil/transplant.py --only=2

Kept as the run ``transplant`` of the episode (``outputs/frontier_lab/fullceil/runs/``).
"""

import pickle
import time

import common as K
import fire
import numpy as np
from descend import milp_path, run_path
from milp import best_run


def main(only: str | tuple | int = "2", task: str = "full", entropy: int = 444, among: str = "tah0_f", name: str = "milp",
         iters: int = 60, hull: int = 3, search: int = 8, min_gain: float = 1e9, both: bool = False) -> None:
    """``both``: also with the plan's whole weeks of the last 12 weeks asked for."""
    import core  # regime_lab's

    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)]
    for n in ns:
        path = run_path("transplant", task, entropy, n)
        if path.is_file():
            continue
        t0 = time.process_time()
        ep = core.Episode.of(task, entropy, n)
        ybar = np.asarray(ep.marks.y_bar, dtype=float)
        F = [gi for gi, fabs in enumerate(ep.inst.grid_fabs) if fabs]
        mp = pickle.loads(milp_path(task, entropy, n, name).read_bytes())
        pshed = mp["plan"]["shed"]
        label, run = best_run(task, entropy, n, among)
        recs0, J0 = ep.simulate(run["acts"])
        shed0 = np.array([r.shed for r in recs0])
        whole_p, short_p = pshed < 5e-4 * ybar, pshed > 5e-3 * ybar
        whole_0, short_0 = shed0 < 5e-4 * ybar, shed0 > 5e-3 * ybar
        print(f"ep {n}: best executed '{label}' {J0 / K.BN:.2f} bn; integer plan own value {mp['J'] / 1e9:.2f} bn. Grid-weeks of grids with fabs: "
              f"plan whole {int(whole_p[:, F].sum())}, executed whole {int(whole_0[:, F].sum())}; plan whole and executed short "
              f"{int((whole_p & short_0)[:, F].sum())}; plan short and executed whole {int((short_p & whole_0)[:, F].sum())}", flush=True)
        names = [ep.inst.nodes[g].id for g in ep.inst.grids]
        print("   by grid, plan whole / executed whole / only the plan / only the executed: " + ", ".join(
            f"{names[gi]} {int(whole_p[:, gi].sum())}/{int(whole_0[:, gi].sum())}/{int((whole_p & short_0)[:, gi].sum())}/{int((short_p & whole_0)[:, gi].sum())}" for gi in F), flush=True)
        method = "ipm" if ep.N > 4 * core.BIG else "simplex"
        out = {"J0": int(J0), "tries": {}}
        best = None
        for until in (ep.T - 12, ep.T) if both else (ep.T - 12,):
            mode, ref = ep.regimes(recs0)
            asked = 0
            for (t, gi), gm in list(mode["grid"].items()):
                if gi in F and gm == "OFF" and t <= until and whole_p[t - 1, gi]:
                    mode["grid"][(t, gi)] = "SOFT"
                    asked += 1
            sol = ep.solve(ep.cell(mode, ref), method=method)
            if sol["status"] != "Optimal":
                print(f"   whole weeks of the plan up to week {until}: {asked} asked, cell {sol['status']}", flush=True)
                continue
            a2 = ep.actions(sol["x"])
            r2, J2 = ep.simulate(a2)
            closed = sum(1 for (t, gi), gm in mode["grid"].items() if gm == "SOFT" and r2[t - 1].shed[gi] < 5e-4 * ybar[t - 1, gi])
            print(f"   whole weeks of the plan up to week {until}: {asked} asked, {closed} closed; played {J2 / K.BN:.2f} bn "
                  f"({(J2 - J0) / K.BN:+.2f} to the best executed)", flush=True)
            d = core.descend(ep, a2, iters, min_gain=min_gain, played=(r2, J2), method=method)
            for _ in range(hull):
                d2 = core.descend(ep, d["acts"], iters=iters, min_gain=min_gain, hull="round", close_until=ep.T - 12, method=method, search=search)
                if d2["J"] > d["J"] - 1e8:
                    break
                d = {**d2, "J0": d["J0"], "hist": d["hist"] + d2["hist"]}
            print(f"   after the descent: {d['J'] / K.BN:.2f} bn ({(d['J'] - J0) / K.BN:+.2f} to the best executed), {time.process_time() - t0:.0f} s CPU", flush=True)
            out["tries"][until] = {"asked": asked, "closed": closed, "J_cell": int(J2), "J": int(d["J"])}
            if best is None or d["J"] < best["J"]:
                best = d
        if best is not None:
            out.update(J=int(best["J"]), acts=best["acts"], hist=best["hist"], done=True, stages=[], start=f"transplant of {name} into {label}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))


if __name__ == "__main__":
    fire.Fire(main)
