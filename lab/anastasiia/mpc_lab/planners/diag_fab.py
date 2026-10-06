import sys
import numpy as np
sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
import planner_D as P
from shockbench_flow.dynamics.env import Env
from shockbench_flow.policies import lp_common as L
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed
n = int(sys.argv[1]); flags = tuple(sys.argv[2].split(",")); tl = int(sys.argv[3])
out, (m, x) = P.solve(n, flags, tl)
inst, omega, marks, fb = P.world(n)
nc, T, tm = m.meta["nc"], m.T, m.meta["template"]
env = Env(fallback=fb)
obs, info = env.reset(inst, "standard", omega, _policy_seed(P.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="d")
done, t = False, 0
while not done:
    xt = np.zeros_like(m.lb); xt[:nc] = x[t * nc:(t + 1) * nc]
    obs, r, done, tr, inf = env.step(L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t]))); t += 1
R = env.trajectory.records
c = lambda tag, t, *rest: float(x[(t - 1) * nc + tm[(tag, *rest)]])
K = [k.id for k in inst.commodities]; wafer = K.index("wafer")
print("gap", out.get("gap"))
for fo, node in enumerate(inst.fabs):
    fa = inst.nodes[node].fab; s = inst.slot_index[(node, wafer)]
    go = inst.grid_ordinal.get(inst.node_index[fa.grid]) if isinstance(fa.grid, str) else (inst.grid_ordinal.get(fa.grid) if fa.grid is not None else None)
    print(f"== fab {fo} {inst.nodes[node].id} cap0 {fa.cap0:.0f} grid {fa.grid}")
    print("week | lots plan / sim | wafer stock end plan / sim | energy plan / sim | cap alpha*R*cap0")
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 40):
        cap = marks.alpha_bar[t - 1, fo] * marks.R[t - 1, fo] * fa.cap0
        print(f"{t:4d} | {c('p', t, fo):9.0f} {R[t-1].lots_started[fo]:9.0f} | {c('I', t, s):10.0f} {R[t-1].stock[s]:10.0f} | {c('E', t, fo):7.1f} {R[t-1].energy[fo]:7.1f} | {cap:9.0f}")
