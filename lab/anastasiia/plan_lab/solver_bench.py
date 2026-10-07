"""HiGHS settings timed on the cells an agent really solves.

    uv run python lab/anastasiia/plan_lab/solver_bench.py grab outputs/plan_lab/agents/hullr_2031 --weeks=5,15,25
    uv run python lab/anastasiia/plan_lab/solver_bench.py bench outputs/plan_lab/agents/hullr_2031

``grab`` plays one episode with an agent built by ``lab/anastasiia/regime_lab/build.py`` and keeps the linear programs
of the named weeks (every solve of those weeks) in ``outputs/plan_lab/20261007_speed/``. ``bench`` solves each of
them under several settings of SciPy's bundled HiGHS and prints, per setting: CPU seconds (mean, maximum), the
objective's distance from the reference setting's, the largest bound or row violation of the solution.
"""

import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


OUT = Path("outputs/plan_lab/20261007_speed")
SETTINGS = {
    "ipm (as now)": {"solver": "ipm"},
    "ipm, no crossover": {"solver": "ipm", "run_crossover": "off"},
    "simplex, cold": {"solver": "simplex"},
    "simplex, warm": {"solver": "simplex", "warm": True},
    "simplex, cold, 1 s then ipm": {"solver": "simplex", "time_limit": 1.0, "then": "ipm"},
}


def grab(agent: str, task: str = "full", entropy: int = 444, episode: int = 1, weeks: str | tuple = "5,15,25") -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import sbf_starter.agents as agents_mod
    from sbf_starter import env_id
    from sbf_starter.agents import resolve

    want = {int(x) for x in (weeks.split(",") if isinstance(weeks, str) else weeks)}
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    folder = resolve(agent)
    ag = agents_mod.load(str(folder))(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
    core = sys.modules[f"{folder.name}_core"]
    solve, kept, now, last = core.Episode.solve, [], [0], [None, 0]  # last: a basis and the week it was found in

    def keeping(self, C, *args, **kwargs):
        out = solve(self, C, *args, **kwargs)
        warm = None  # the last cell's basis, moved on by a week when that cell was last week's
        if last[0] is not None:
            warm = last[0] if last[1] == now[0] else self.shifted(last[0], now[0] - last[1])
        if now[0] in want:
            A, lo, hi = self.rows(C)
            kept.append({"week": now[0], "obj": np.array(self.obj if C.cost is None else self.obj + C.cost),
                         "lb": np.array(C.lb), "ub": np.array(C.ub), "A": A.tocsc(), "lo": np.array(lo),
                         "hi": np.array(hi), "offset": float(self.offset), "J": out.get("J"),
                         "seconds": out.get("seconds"), "status": out.get("status"), "warm": warm,
                         "first": last[1] != now[0]})
        if out.get("basis") is not None:
            last[0], last[1] = out["basis"], now[0]
        return out

    core.Episode.solve = keeping
    for week in range(1, max(want) + 1):
        now[0] = week
        obs, _reward, term, trunc, info = env.step(ag.act(obs))
        if term or trunc:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cells_{folder.name}_{task}_{entropy}_{episode}.pkl"
    path.write_bytes(pickle.dumps(kept))
    print(f"{len(kept)} cells of weeks {sorted(want)} in {path}: columns {[c['A'].shape[1] for c in kept]}, rows "
          f"{[c['A'].shape[0] for c in kept]}, HiGHS's own seconds {[round(c['seconds'], 2) for c in kept]}")


def bench(agent: str, task: str = "full", entropy: int = 444, episode: int = 1, only: str = "") -> None:
    from scipy.optimize._highspy import _core as hs  # the build SciPy ships

    folder = Path(agent)
    cells = pickle.loads((OUT / f"cells_{folder.name}_{task}_{entropy}_{episode}.pkl").read_bytes())
    inf = hs.kHighsInf
    print(f"{len(cells)} cells; columns {cells[0]['A'].shape[1]}, rows {cells[0]['A'].shape[0]}, nonzeros {cells[0]['A'].nnz}")
    print(f"{'setting':28} {'CPU s mean':>10} {'max':>6} {'J - reference, M USD (max abs)':>32} {'worst violation':>16}  statuses")
    ref = {}
    for name, options in SETTINGS.items():
        if only and only not in name:
            continue
        secs, gaps, viol, statuses = [], [], [], set()
        for i, c in enumerate(cells):
            lp = hs.HighsLp()
            n_row, n_col = c["A"].shape
            lp.num_col_, lp.num_row_ = n_col, n_row
            lp.col_cost_, lp.col_lower_ = c["obj"], c["lb"]
            lp.col_upper_ = np.where(np.isinf(c["ub"]), inf, c["ub"])
            lp.row_lower_, lp.row_upper_ = np.where(np.isinf(c["lo"]), -inf, c["lo"]), np.where(np.isinf(c["hi"]), inf, c["hi"])
            lp.offset_ = c["offset"]
            lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
            lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
            lp.a_matrix_.start_, lp.a_matrix_.index_ = c["A"].indptr.astype(np.int32), c["A"].indices.astype(np.int32)
            lp.a_matrix_.value_ = c["A"].data.astype(np.float64)
            t0 = time.process_time()
            for attempt in (options, {"solver": options.get("then")}):
                h = hs._Highs()
                h.setOptionValue("output_flag", False)
                h.setOptionValue("time_limit", 60.0)
                for key, value in attempt.items():
                    if key not in ("warm", "then"):
                        h.setOptionValue(key, value)
                h.passModel(lp)
                warm = c.get("warm") if attempt.get("warm") else None
                if warm is not None and len(warm[0]) == n_col and len(warm[1]) == n_row:
                    codes = {int(v): v for v in (hs.HighsBasisStatus.kLower, hs.HighsBasisStatus.kBasic, hs.HighsBasisStatus.kUpper,
                                                 hs.HighsBasisStatus.kZero, hs.HighsBasisStatus.kNonbasic)}
                    b = hs.HighsBasis()
                    b.col_status, b.row_status = [codes[int(v)] for v in warm[0]], [codes[int(v)] for v in warm[1]]
                    b.valid, b.alien = True, True
                    h.setBasis(b)
                h.run()
                status = h.modelStatusToString(h.getModelStatus())
                if status == "Optimal" or not options.get("then") or attempt is not options:
                    break
            secs.append(time.process_time() - t0)
            statuses.add(status)
            x = np.asarray(h.getSolution().col_value, dtype=float)
            if len(x) == n_col:
                J = float(c["obj"] @ x) + c["offset"]
                ref.setdefault(i, J)
                gaps.append(abs(J - ref[i]) / 1e6)
                rows = c["A"] @ x
                scale = 1.0
                viol.append(max(float(np.max(c["lb"] - x)), float(np.max(x - c["ub"])),
                                float(np.max(c["lo"] - rows)), float(np.max(rows - c["hi"])), 0.0) / scale)
        firsts = [x for x, c in zip(secs, cells) if c.get("first", True)]
        seconds = [x for x, c in zip(secs, cells) if not c.get("first", True)]
        print(f"{name:28} {np.mean(secs):10.2f} {np.max(secs):6.2f} {max(gaps) if gaps else float('nan'):32.3f} "
              f"{max(viol) if viol else float('nan'):16.2e}  {sorted(statuses)}; p95 {np.percentile(secs, 95):.2f}; "
              f"first solve of a week {np.mean(firsts) if firsts else float('nan'):.2f}, second {np.mean(seconds) if seconds else float('nan'):.2f}")


if __name__ == "__main__":
    fire.Fire({"grab": grab, "bench": bench})
