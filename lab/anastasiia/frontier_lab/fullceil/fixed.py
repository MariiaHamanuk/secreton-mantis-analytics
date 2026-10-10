"""What the integer program's freed rules are worth at the executed plan's own whole weeks (notes/u_fullceil.md): the
program "base load first" with every binary fixed where the best executed plan has it, solved as a linear program.

    uv run python lab/anastasiia/frontier_lab/fullceil/fixed.py --only=2,3,9

The executed plan's played cost minus this value is what the simulator's other automatic rules (fuel burnt in
proportion to the load, every wafer started, everything packaged, straits oldest first) cost at the same set of whole
weeks; the rest of the distance to the integer program's own value is its other set of weeks. A program's value, not
a played cost. Kept in ``outputs/frontier_lab/fullceil/milp/fixed_<task>_<root>_<episode>.pkl``.
"""

import pickle
import time

import common as K
import fire
import numpy as np
from descend import milp_path
from milp import best_run, lifted, program


def main(only: str | tuple | int = "", task: str = "full", entropy: int = 444, first: int = 0, episodes: int = 16,
         among: str = "tah0_f") -> None:
    import core  # regime_lab's
    import highspy
    from shockbench_flow.policies import lp_common as L

    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else range(first, first + episodes)
    for n in ns:
        path = milp_path(task, entropy, n, "fixed")
        got = best_run(task, entropy, n, among)
        if path.is_file() or got is None:
            continue
        label, run = got
        t0 = time.process_time()
        ep = core.Episode.of(task, entropy, n)
        recs, J_exec = ep.simulate(run["acts"])
        _mode, ref = ep.regimes(recs)
        P = program(ep)
        c, lb, ub, A, rl, ru, off, ncol = P
        full, _J, _viol, _rel = lifted(ep, P, ep.vector(recs, ref)[: ep.n0])
        lb, ub = lb.copy(), ub.copy()
        lb[ncol:] = ub[ncol:] = full[ncol:]
        h = highspy.Highs()
        for key, val in (("output_flag", False), ("threads", 1), ("solver", "ipm"), ("time_limit", 600.0)):
            h.setOptionValue(key, val)
        h.passModel(L.highs_lp(c, lb, ub, A, rl, ru, off))
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        J = float(h.getInfo().objective_function_value)
        print(f"ep {n}: best executed '{label}' played {J_exec / K.BN:.2f} bn; the program at the same whole weeks ({int(full[ncol:].sum())} "
              f"grid-weeks whole) {J / 1e9:.2f} bn ({status}): the freed rules are worth {J_exec / K.BN - J / 1e9:.2f} bn there; "
              f"{time.process_time() - t0:.0f} s CPU", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"J": J, "status": status, "J_exec": int(J_exec), "start": label, "whole": int(np.sum(full[ncol:]))}))


if __name__ == "__main__":
    fire.Fire(main)
