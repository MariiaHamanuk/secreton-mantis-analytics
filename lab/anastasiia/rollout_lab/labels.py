"""The model as a teacher: what a change of one grid's fuel is worth this week, and the state it was made in.

    uv run python lab/anastasiia/rollout_lab/labels.py --task=small --episodes=16 --out=labels_small.json
    uv run python lab/anastasiia/rollout_lab/labels.py --task=full --episodes=6 --every=8 --out=labels_full.json

A rule agent plays real episodes. Every ``every`` weeks, for every grid and fuel, its copy plays ``weeks`` weeks on
``model.Model`` with the true network of those weeks (the environment's own dynamics, see ``check.py``): once as it
is (the base), and once per change of this week's action, the copy playing its own rules afterwards:

- ``order+``: one more week of the fuel's full burn ordered for the grid, spread over its open lanes;
- ``order-``: half of what the rules ordered for it;
- ``valve+``: half a week's burn more moved from the terminal to the grid;
- ``valve0``: nothing moved from the terminal to the grid.

A row per (decision, grid, fuel): the state as the fuel rules saw it (their own trace of the week: mode, stocks at
the grid and the terminal, cargo on the way, the inventory position, what they ordered and moved, the weekly burn) and
each change's score: its cost minus the base's over 8, 16 and all the weeks, and the shed and unserved-demand parts of
the last, in bn USD; negative means the change pays. A change that does nothing in that state has no score.

The labels know the future (the true network), the state does not: a rule read off them is tested as any rule is,
by a paired comparison of agents. A label is one step around the base rules, so labels of different weeks do not add up.
"""

import copy
import json
from pathlib import Path

import check
import fire
import numpy as np
from joblib import Parallel, delayed


CHANGES = ("order+", "order-", "valve+", "valve0")
CUTS = (8, 16)  # the horizons kept beside the whole rollout
BN = 1e9


def changed(action, change: str, fuel, g: int, k: int, mask: np.ndarray, burn: float):
    """``action`` with ``change`` applied to grid g's fuel k; None when the change does nothing there."""
    flows = np.array(action["flows"], dtype=float)
    if change == "order+":
        slots = [s for s in fuel.ship.get((g, k), []) if mask[s]]
        if not slots:
            return None
        flows[slots] += burn / len(slots)
    elif change == "order-":
        slots = list(fuel.ship.get((g, k), []))
        if flows[slots].sum() <= 0:
            return None
        flows[slots] *= 0.5
    elif change == "valve+":
        slots = [s for s in fuel.tg.get((g, k), []) if mask[s]]
        if not slots:
            return None
        flows[slots[0]] += 0.5 * burn
    else:
        slots = list(fuel.tg.get((g, k), []))
        if flows[slots].sum() <= 0:
            return None
        flows[slots] = 0.0
    return dict(action) | {"flows": flows}


def episode(agent: str, task: str, entropy: int, n: int, weeks: int, every: int, changes: tuple) -> list:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    lp_part, model = check._modules()
    from sbfv.marks import FabHit

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ep = u.core._ep
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    player = load(agent)(config)
    planner = lp_part.Planner(config)
    m = model.Model(config, planner)
    inst = m.inst
    truth = {name: np.asarray(getattr(ep.marks, name)) for name in model.L.WINDOW_FIELDS}
    fuel = player.fuel
    names = [c.id for c in inst.commodities]
    node = [x.id for x in inst.nodes]
    pairs = sorted({*fuel.ship, *fuel.tg})

    def roll(first, arrays, hits, edit) -> np.ndarray | None:
        """(weeks, [all, shed, unserved]) costs of a rollout whose first action is ``edit``-ed; None: no change."""
        policy, w, seen, spent = copy.deepcopy(player), m.window(first, weeks, arrays, hits), first, []
        for h in range(weeks):
            action = policy.act(seen)
            if h == 0 and edit is not None:
                action = edit(action)
                if action is None:
                    return None
            costs = m.step(w, action).costs
            spent.append((costs.total(), costs.shed, costs.shortage))
            seen = m.flat(first, w)
        return np.array(spent)

    rows, done = [], False
    while not done:
        week = int(obs["week"][0])
        planner._remember(obs)
        if (week - 1) % every == 0 and week + weeks - 1 <= inst.T:
            real = {name: a[week - 1 : week - 1 + weeks] for name, a in truth.items()}
            hits = [
                FabHit(h.fab, h.onset - (week - 1), h.severity)
                for h in ep.marks.fab_hits
                if week <= h.onset_week < week + weeks
            ]
            probe = copy.deepcopy(player)  # the rules' own account of the week, without touching the player
            probe.fuel.P = dict(probe.fuel.P) | {"trace": True}
            probe.fuel.trace = []
            probe.act(obs)
            seen_by_rules = {(r["grid"], r["k"]): r for r in probe.fuel.trace if r["week"] == week}
            mask = np.asarray(obs["action_mask"]) == 1
            last_shed = np.asarray(obs["last_week.shed.qty"], dtype=float)
            output = np.asarray(obs["graph_now.grid.G_bar"], dtype=float)
            base = roll(obs, real, hits, None)
            for g, k in pairs:
                gd = fuel.grids[g]
                burn = gd["fuels"][k] * gd["deliverable"]
                role = "rationed" if k == gd["rationed"] else "stocked" if (g, k) in fuel.stocked else "plain"
                row = {
                    "episode": n,
                    "week": week,
                    "left": inst.T - week + 1,
                    "grid": node[g],
                    "fuel": names[k],
                    "role": "gate" if (g, k) in fuel.gate else role,
                    "burn": burn,
                    "premium": fuel.premium[g],
                    "sliver": fuel.sliver[g] / gd["deliverable"],
                    "output": float(output[fuel.grid_row[g]] / gd["deliverable"]),
                    "shed_last": float(last_shed[fuel.grid_row[g]] / gd["deliverable"]),
                    "lanes": len(fuel.ship.get((g, k), [])),
                    "lanes_open": int(sum(mask[s] for s in fuel.ship.get((g, k), []))),
                    "rules": {a: b for a, b in seen_by_rules.get((g, k), {}).items() if a not in ("week", "grid", "k")},
                }
                for change in changes:
                    spent = roll(
                        obs, real, hits, lambda a, c=change, g=g, k=k, b=burn: changed(a, c, fuel, g, k, mask, b)
                    )
                    if spent is None:
                        continue
                    gap = (spent - base) / BN
                    row[change] = {
                        **{f"w{cut}": float(gap[:cut, 0].sum()) for cut in CUTS if cut < weeks},
                        "all": float(gap[:, 0].sum()),
                        "shed": float(gap[:, 1].sum()),
                        "unserved": float(gap[:, 2].sum()),
                    }
                rows.append(row)
        obs, _reward, term, trunc, _info = env.step(player.act(obs))
        done = term or trunc
    return rows


def main(
    agent: str = "agents/anastasiia_rules_v3",
    task: str = "small",
    entropy: int = 111,
    episodes: int = 4,
    first: int = 0,
    weeks: int = 24,
    every: int = 4,
    changes: str = ",".join(CHANGES),
    n_jobs: int = 2,
    out: str = "",
) -> None:
    """Play the rollouts and keep a row per (decision, grid, fuel) (see the module docstring).

    Args:
        agent: a rule agent (it must keep ``fuel``, a ``FuelRules`` with its trace), a folder or a name of agents/.
        task: small or full.
        entropy: the root of the episodes.
        episodes: how many episodes, from ``first``.
        first: the first episode.
        weeks: weeks of a rollout.
        every: a decision is labelled every this many weeks.
        changes: the changes to try, of ``CHANGES``, comma-separated.
        n_jobs: worker processes.
        out: a JSON file for the rows.

    """
    wanted = tuple(c for c in (changes if isinstance(changes, (tuple, list)) else str(changes).split(",")) if c)
    runs = Parallel(n_jobs=n_jobs)(
        delayed(episode)(agent, task, entropy, n, weeks, every, wanted) for n in range(first, first + episodes)
    )
    rows = [row for run in runs for row in run]
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows))
    print(f"{agent} on {task}, episodes {first}..{first + episodes - 1} of root {entropy}: {len(rows)} rows")
    print("mean score over the whole rollout, bn USD per decision (negative: the change pays), and how often it pays")
    table = []
    for role in sorted({r["role"] for r in rows}):
        for change in wanted:
            got = np.array([r[change]["all"] for r in rows if r["role"] == role and change in r])
            if len(got):
                table.append(
                    [role, change, len(got), float(got.mean()), float(np.mean(got < -1e-6)), float(np.median(got))]
                )
    check.table(["the fuel's role", "change", "cases", "mean", "pays in", "median"], table)


if __name__ == "__main__":
    fire.Fire(main)
