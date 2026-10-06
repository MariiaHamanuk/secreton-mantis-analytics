"""Where an agent loses to the clairvoyant plan: its cost by component, by harm level and by week, beside the naive
rule's and the clairvoyant plan's on the same scenarios, and the routes on which the clairvoyant plan ships differently.

    uv run python lab/anastasiia/stats_lab/cost_gap.py
    uv run python lab/anastasiia/stats_lab/cost_gap.py --agent=template --task=small --episodes=64

Each episode is played once by the agent (its weekly cost components and executed flows), once by the naive rule, and
solved once as the clairvoyant plan's linear program (its weekly components and flows). The totals are checked against
the cached references. The agent plays under gymnasium with policy seed 0, so an agent that draws random numbers gets
other draws than under ``sbf evaluate``.

Writes ``summary.md`` and ``episodes.npz`` under ``outputs/cost_gap/<date_time>/``.
"""

import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


def episode(agent: str, task: str, entropy: int, n: int) -> dict:
    """One episode's weekly cost components of the agent, the naive rule and the clairvoyant plan, and their flows."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.dynamics.state import COST_COMPONENTS
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp, lp_costs, lp_flow_key, solve_oracle
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    record = record_episode(env, load(agent), options={"episode": n}, naive_replications=REPLICATIONS)
    inst, params = task_generator(task)
    omega = sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train")
    model = build_lp(inst, compute_marks(inst, omega))
    plan = solve_oracle(model)
    weekly, credit = lp_costs(model, plan.x)
    slots = record["static"]["action_slots"]
    best = np.zeros((inst.T, len(slots["edge"])))
    for s, (e, k, lane) in enumerate(zip(slots["edge"], slots["k"], slots["lane"])):
        for t in range(1, inst.T + 1):
            j = model.index.get(("x", t, *lp_flow_key(inst, e, k, lane)))
            if j is not None:
                best[t - 1, s] = plan.x[j]
    obs = record["obs"]
    naive = record["naive"]
    return {
        "agent": record["costs"],  # (T, 8) USD
        "agent_credit": record["meta"]["salvage_cents"] / 100,
        "agent_J": record["meta"]["J_cents"] / 100,
        "naive": naive["costs"],
        "naive_J": int(record["meta"]["naive_J_cents"]) / 100,
        "best": np.array([[w.as_dict()[c] for c in COST_COMPONENTS] for w in weekly]),
        "best_credit": credit,
        "best_J": plan.J_cents / 100,
        "asked": obs["last_week.clip.requested"][1:],  # (T, slots)
        "sent": obs["last_week.clip.executed"][1:],
        "best_sent": best,
        "capacity": obs["graph_now.u"][:-1][:, slots["edge"]],
    }


def table(header: list[str], rows: list[list]) -> str:
    def cell(x) -> str:
        return f"{x:,.2f}" if isinstance(x, float) else str(x)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]) + "\n"


def report(agent: str, task: str, entropy: int, eps: list[dict], refs: list[dict], static: dict) -> str:
    from shockbench_flow.dynamics.state import COST_COMPONENTS

    n_ep = len(eps)
    T = eps[0]["agent"].shape[0]
    stack = {k: np.stack([np.asarray(e[k], dtype=float) for e in eps]) for k in eps[0]}
    scale = 1e9
    out = [f"# Where `{agent}` loses to the clairvoyant plan: {task}, {n_ep} episodes of root {entropy}\n"]
    out.append(
        "USD billions per episode, means over the episodes. Written by `lab/anastasiia/stats_lab/cost_gap.py`.\n"
    )

    drift = max(
        abs(e[f"{who}_J"] - r[key] / 100)
        for e, r in zip(eps, refs)
        for who, key in (("naive", "J_naive_cents"), ("best", "J_oracle_cents"))
        if r[key] is not None
    )
    out.append(f"Check against the cached references: the largest difference of an episode's cost is ${drift:,.2f}.\n")

    # 1. by component
    out.append("## 1. By cost component\n")
    gap_total = float((stack["agent_J"] - stack["best_J"]).mean())
    rows = []
    for c, name in enumerate(COST_COMPONENTS):
        a, b, v = (float(stack[who][:, :, c].sum(axis=1).mean()) for who in ("naive", "agent", "best"))
        rows.append([name, a / scale, b / scale, v / scale, (b - v) / scale, 100 * (b - v) / gap_total])
    a, b, v = (
        -float(stack[f"{who}_credit"].mean()) if who != "naive" else float("nan") for who in ("naive", "agent", "best")
    )
    a = float((stack["naive_J"] - stack["naive"].sum(axis=(1, 2))).mean())
    rows.append(["end stock credit", a / scale, b / scale, v / scale, (b - v) / scale, 100 * (b - v) / gap_total])
    a, b, v = (float(stack[f"{who}_J"].mean()) for who in ("naive", "agent", "best"))
    rows.append(["**total**", a / scale, b / scale, v / scale, (b - v) / scale, 100.0])
    head = ["component", "naive", agent, "clairvoyant", f"{agent} - clairvoyant", "share of the gap %"]
    out.append(table(head, rows))
    out.append(f"Score on these episodes, unweighted: {(a - b) / (a - v):.4f}.\n")

    # 2. by harm level
    out.append("## 2. By harm level\n")
    level = np.array([r.get("stratum") or 0 for r in refs])
    rows = []
    for s in sorted(set(level.tolist())):
        m = level == s
        a, b, v = (float(stack[f"{who}_J"][m].mean()) for who in ("naive", "agent", "best"))
        parts = (stack["agent"][m] - stack["best"][m]).sum(axis=1).mean(axis=0)
        top = np.argsort(-np.abs(parts))[:3]
        text = ", ".join(f"{COST_COMPONENTS[c]} {parts[c] / scale:+,.1f}" for c in top)
        rows.append([s, int(m.sum()), (a - v) / scale, (b - v) / scale, (a - b) / (a - v), text])
    head = ["level", "episodes", "naive - clairvoyant", f"{agent} - clairvoyant", "score", "largest parts of the gap"]
    out.append(table(head, rows))

    # 3. by week
    out.append("## 3. By week\n")
    out.append("The weekly cost without the end stock credit; a week's gap can be negative (the plan pays early).\n")
    edges = np.linspace(0, T, 5).astype(int)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b, v = (float(stack[who][:, lo:hi].sum(axis=(1, 2)).mean()) for who in ("naive", "agent", "best"))
        parts = (stack["agent"][:, lo:hi] - stack["best"][:, lo:hi]).sum(axis=1).mean(axis=0)
        top = np.argsort(-np.abs(parts))[:3]
        text = ", ".join(f"{COST_COMPONENTS[c]} {parts[c] / scale:+,.1f}" for c in top)
        rows.append([f"{lo + 1}-{hi}", a / scale, b / scale, v / scale, (b - v) / scale, text])
    head = ["weeks", "naive", agent, "clairvoyant", f"{agent} - clairvoyant", "largest parts of the gap"]
    out.append(table(head, rows))

    # 4. flows
    out.append("## 4. What the agent asks for and what moves\n")
    asked, sent, best, cap = (stack[k] for k in ("asked", "sent", "best_sent", "capacity"))
    edge_id, mode = static["edges"]["id"], static["edges"]["mode"]
    goods = static["commodities"]["id"]
    slots = static["action_slots"]
    out.append(
        f"Of the volume `{agent}` asks for, {100 * float(sent.sum() / asked.sum()):.0f}% is executed; the rest is cut\n"
        "by the week's capacity, a closed strait or missing stock.\n"
    )
    rows = []
    for k, name in enumerate(goods):
        m = np.array(slots["k"]) == k
        if m.any() and best[:, :, m].sum() + sent[:, :, m].sum() > 0:
            rows.append(
                [name, float(asked[:, :, m].sum() / n_ep / T), float(sent[:, :, m].sum() / n_ep / T)]
                + [float(best[:, :, m].sum() / n_ep / T), float(sent[:, :, m].sum() / max(best[:, :, m].sum(), 1e-9))]
            )
    head = ["commodity", "asked / week", "executed / week", "clairvoyant / week", "executed / clairvoyant"]
    out.append("### By commodity (dispatch slots only, units of the commodity)\n\n" + table(head, rows))
    rows = []
    for name in sorted(set(mode[e] for e in slots["edge"])):
        m = np.array([mode[e] == name for e in slots["edge"]])
        share = [float((x[:, :, m] / np.maximum(cap[:, :, m], 1e-9)).mean()) for x in (sent, best)]
        rows.append([name, int(m.sum()), 100 * share[0], 100 * share[1]])
    head = ["mode", "slots", f"{agent}: mean use of the week's capacity %", "clairvoyant %"]
    out.append("### By transport mode\n\n" + table(head, rows))

    use_a = (sent / np.maximum(cap, 1e-9)).mean(axis=(0, 1))
    use_b = (best / np.maximum(cap, 1e-9)).mean(axis=(0, 1))
    order = np.argsort(use_b - use_a)
    for title, picks in (("ships less than", order[:15]), ("ships more than", order[::-1][:15])):
        rows = []
        for s in picks:
            lane = slots["lane"][s]
            where = edge_id[slots["edge"][s]] + ("" if lane is None else f" (lane {static['lanes']['id'][lane]})")
            rows.append([where, goods[slots["k"][s]], 100 * float(use_a[s]), 100 * float(use_b[s])])
        head = ["slot", "commodity", f"{agent}: use of capacity %", "clairvoyant %"]
        out.append(f"### Slots where the clairvoyant plan {title} `{agent}`\n\n" + table(head, rows))
    return "\n".join(out)


def main(
    agent: str = "template",
    task: str = "tiny",
    episodes: int = 16,
    entropy: int = 111,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """Compare ``agent`` with the naive rule and the clairvoyant plan, cost component by cost component.

    Args:
        agent: an agent's name (a folder of agents/) or a submission folder.
        task: tiny, small or full.
        episodes: episodes 0 .. episodes - 1 of the root.
        entropy: the scenarios' root (111, the team's tuning root).
        n_jobs: workers (-1: all cores).
        out: the folder to write to (default: outputs/cost_gap/<date_time>/).

    """
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments

    from sbf_starter import ROOT, env_id, scoring

    folder = Path(out) if out else ROOT / "outputs" / "cost_gap" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)  # fills the cache too
    start = time.perf_counter()
    eps = Parallel(n_jobs=n_jobs)(delayed(episode)(agent, task, entropy, n) for n in range(episodes))
    print(f"{episodes} episodes of {task} played and solved in {time.perf_counter() - start:.0f} s")
    _, info = gym.make(env_id(task), entropy=entropy).reset(options={"episode": 0})
    (folder / "summary.md").write_text(report(agent, task, entropy, eps, refs, info["static"]))
    np.savez_compressed(folder / "episodes.npz", **{k: np.stack([np.asarray(e[k]) for e in eps]) for k in eps[0]})
    print(f"written {folder}")


if __name__ == "__main__":
    fire.Fire(main)
