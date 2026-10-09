"""Play the first weeks of one episode with an agent folder and print its weekly notes and one grid's weeks.

    uv run python lab/anastasiia/regime_lab/probe.py outputs/regime_lab/agents/plan_L_hull2 --task=full --episode=2 --weeks=40 --grid=grid_jp

Per week: the agent's note (reference, taken or kept, whole weeks written), CPU seconds, and for the grid: its shed
share of the base load, each fuel segment's burn as a share of its cap, the stock at the grid and at its terminal,
and the lots its fabs started as a share of capacity.
"""

import sys
import time
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402, F401


def main(agent: str, task: str = "full", entropy: int = 444, episode: int = 0, weeks: int = 40, grid: str = "grid_jp") -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    u = env.unwrapped
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    if hasattr(ag, "tell_truth"):
        ag.tell_truth({"marks": u.core._ep.marks})
    inst, marks = u.core._ep.inst, u.core._ep.marks
    name = [nd.id for nd in inst.nodes]
    com = [c.id for c in inst.commodities]
    g = name.index(grid)
    gi = list(inst.grids).index(g)
    ga = inst.nodes[g].grid
    fabs = [fi for fi in inst.grid_fabs[gi]]
    term = [inst.edges[e].tail for e in inst.in_edges[g] if name[inst.edges[e].tail].startswith("term")]
    slot = {(s.node, s.k): i for i, s in enumerate(inst.stock_slots)}
    for week in range(1, weeks + 1):
        t0 = time.process_time()
        action = ag.act(obs)
        cpu = time.process_time() - t0
        obs, _r, term_, trunc, _i = env.step(action)
        rec = u.core._ep.traj.records[-1]
        burn = " ".join(f"{com[k]} {rec.segment[(gi, k)] / max(1e-9, ga.shares[k] * float(marks.G_bar[week - 1][gi])):.2f}" for k in ga.fuels)
        at = " ".join(f"{com[k]} {rec.stock[slot[(g, k)]]:.0f}" + "".join(f"/{rec.stock[slot[(x, k)]]:.0f}" for x in term if (x, k) in slot) for k in ga.fuels)
        lots = " ".join(f"{rec.lots_started[fi] / inst.nodes[inst.fabs[fi]].fab.cap0:.2f}" for fi in fabs)
        note = ag.log[-1] if getattr(ag, "log", None) else ()
        print(f"week {week:3d} cpu {cpu:5.2f} shed {rec.shed[gi] / float(marks.y_bar[week - 1][gi]) * 100:5.1f}%  burn {burn}  stock grid/terminal {at}  lots {lots}  {note[1:] if note else ''}", flush=True)
        if term_ or trunc:
            break


if __name__ == "__main__":
    fire.Fire(main)
