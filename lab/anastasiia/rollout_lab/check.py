"""Does the model inside the agent say what the environment then does? One week ahead, and a rollout of several.

    uv run python lab/anastasiia/rollout_lab/check.py --task=small --episodes=4
    uv run python lab/anastasiia/rollout_lab/check.py --task=full --episodes=2 --rollout=8 --every=8

An agent plays real episodes. Every week, before its action goes to the environment, ``model.Model`` is asked:

1. state: the simulator's state rebuilt from the observation, against the environment's own state;
2. round trip: that state written back as an observation (``Model.flat``), against the observation it came from;
3. one week: the model's record of the week under the agent's action, against the environment's record, twice: with
   the week's true network (what the rebuilding itself costs) and with the network "as observed" (what the agent has);
4. next observation: ``Model.flat`` after that week (true network), against the next real observation;
5. every ``every`` weeks a rollout: a copy of the agent plays ``rollout`` weeks on the model, and the cost the model
   gives those weeks is set against what they then cost in the environment. The simulator's network is "as
   observed", then the true one of those weeks (the copy still reads the network of the first week: what is left
   then is the agent's own view going stale, not the simulator's forecast), then, with ``--by_field``, as observed
   but for one group of fields that is true (which knowledge of the future the forecast lacks).

A difference is the sum of absolute differences over the sum of the true values, over all weeks and episodes
("share off"); "weeks off" is the share of weeks where it exceeds one millionth of the week's true total. Times are
CPU seconds of this machine, not of the scoring container. Read-only; ``--out`` keeps the numbers as JSON.
"""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HYBRID = ROOT / "agents" / "anastasiia_hybrid_chiplp"
RECORD = ("stock", "executed", "lots_started", "energy", "shed", "served", "disposal", "cost")
RECORD += ("cost: shortage", "cost: shed", "cost: the rest")
NEXT = (
    "stock.qty",
    "backlog.qty",
    "queue_lots.qty",
    "pipeline",
    "wip",
    "last_week.clip.executed",
    "last_week.shed.qty",
)
TOL = 1e-6
# the network's fields by what they describe: a rollout is repeated with one group true and the rest as observed
GROUPS = {
    "edge capacity": ("u",),
    "straits": ("o", "kappa", "wr_class", "h_queue", "c_wr"),
    "grids": ("G_bar", "y_bar"),
    "prohibitions": ("prohibited",),
    "tariffs and freight": ("tariff", "c"),
    "supply": ("supply",),
    "fabs and plants": ("R", "alpha_bar", "sigma_scr", "R_osat"),
    "demand": ("demand",),
}


def _modules():
    """The hybrid's ``lp_part`` (it puts ``sbfv`` on the path) and this folder's ``model``."""
    if "rollout_lp_part" not in sys.modules:
        spec = importlib.util.spec_from_file_location("rollout_lp_part", HYBRID / "lp_part.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rollout_lp_part"] = module
        spec.loader.exec_module(module)
        sys.path.insert(0, str(HERE))
    import model

    return sys.modules["rollout_lp_part"], model


class Tally:
    """Sums of |difference| and of |truth| per name, and the weeks in which they differ."""

    def __init__(self):
        self.off, self.total, self.weeks, self.bad = {}, {}, {}, {}

    def add(self, name: str, guess, truth) -> None:
        guess, truth = np.asarray(guess, dtype=float).ravel(), np.asarray(truth, dtype=float).ravel()
        off, total = float(np.abs(guess - truth).sum()), float(np.abs(truth).sum())
        self.off[name] = self.off.get(name, 0.0) + off
        self.total[name] = self.total.get(name, 0.0) + total
        self.weeks[name] = self.weeks.get(name, 0) + 1
        self.bad[name] = self.bad.get(name, 0) + (off > TOL * max(total, 1.0))

    def keyed(self, name: str, guess: dict, truth: dict) -> None:
        keys = sorted(set(guess) | set(truth), key=str)
        self.add(name, [guess.get(k, 0.0) for k in keys], [truth.get(k, 0.0) for k in keys])

    def merge(self, other: "Tally") -> None:
        for mine, theirs in (
            (self.off, other.off),
            (self.total, other.total),
            (self.weeks, other.weeks),
            (self.bad, other.bad),
        ):
            for name, value in theirs.items():
                mine[name] = mine.get(name, 0) + value

    def rows(self, names) -> list[list]:
        return [
            [n, self.off[n] / max(self.total[n], 1e-300), self.bad[n] / max(self.weeks[n], 1), self.weeks[n]]
            for n in names
            if n in self.off
        ]


def _state_dicts(state, shift: int = 0) -> dict:
    """A simulator state as keyed quantities (weeks moved by ``shift``), for comparison."""
    moving, queue, order = {}, {}, {}
    for s in state.pipeline:
        key = (s.edge, s.k, s.lane, s.arrival_week + shift)
        moving[key] = moving.get(key, 0.0) + s.qty
    for lot in state.lots:
        key = (lot.chokepoint, lot.k, lot.lane, lot.next_edge, lot.arrival_week + shift)
        queue[key] = queue.get(key, 0.0) + lot.qty
        key += (lot.dispatch_week + shift, lot.entry_edge)  # what orders the lots an override takes
        order[key] = order.get(key, 0.0) + lot.qty
    fab = {(f, start + shift): q for f, book in state.fab_wip.items() for start, q in book.items() if q}
    plant = {
        (o, w + shift, k): q
        for o, weeks in state.osat_wip.items()
        for w, book in weeks.items()
        for k, q in book.items()
        if q
    }
    return {"pipeline": moving, "queue": queue, "queue order": order, "fab wip": fab, "plant wip": plant}


def _record(rec, n_slots: int) -> dict:
    sent = np.zeros(n_slots)
    for s, q in rec.executed.items():
        sent[s] = q
    return {
        "stock": rec.stock,
        "executed": sent,
        "lots_started": rec.lots_started,
        "energy": rec.energy,
        "shed": rec.shed,
        "served": rec.served,
        "disposal": rec.disposal,
        "cost": [rec.costs.total()],
        "cost: shortage": [rec.costs.shortage],
        "cost: shed": [rec.costs.shed],
        "cost: the rest": [rec.costs.total() - rec.costs.shortage - rec.costs.shed],
    }


def _spent(costs) -> tuple[float, float, float]:
    """A week's cost: all of it, the shed base load, the unserved demand."""
    return costs.total(), costs.shed, costs.shortage


def _listed(obs, block: str, names: tuple) -> dict:
    """A padded block of the flat observation as keyed quantities (order does not matter)."""
    live = np.asarray(obs[f"{block}.qty.observed"]) == 1
    out: dict = {}
    for i in np.flatnonzero(live):
        key = tuple(
            int(obs[f"{block}.{n}"][i]) if obs[f"{block}.{n}.observed"][i] else None for n in names if n != "qty"
        )
        out[key] = out.get(key, 0.0) + float(obs[f"{block}.qty"][i])
    return {k: q for k, q in out.items() if q}


def _next(tally: Tally, prefix: str, guess, truth, model) -> None:
    for name in NEXT:
        if name == "pipeline":
            tally.keyed(prefix + name, _listed(guess, name, model.PIPELINE), _listed(truth, name, model.PIPELINE))
        elif name == "wip":
            tally.keyed(prefix + name, _listed(guess, name, model.WIP), _listed(truth, name, model.WIP))
        else:
            tally.add(prefix + name, guess[name], truth[name])


def episode(
    agent: str, task: str, entropy: int, n: int, rollout: int, every: int, pending: bool, by_field: bool, grids: str
) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    lp_part, model = _modules()
    from sbfv.marks import FabHit

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ep = u.core._ep  # the trusted side's episode: its state, its records, its marks
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    player = load(agent)(config)
    planner = lp_part.Planner(config)
    table = None if grids == "none" else json.loads((HERE / "grid_recovery.json").read_text())
    m = model.Model(config, planner, table, grids)
    fields = model.L.WINDOW_FIELDS
    truth = {name: np.asarray(getattr(ep.marks, name)) for name in fields}
    n_slots, T = len(m.slot_edge), m.inst.T

    def true_network(week: int, weeks: int):
        rows = {name: a[week - 1 : week - 1 + weeks] for name, a in truth.items()}
        hits = [
            FabHit(h.fab, h.onset - (week - 1), h.severity)
            for h in ep.marks.fab_hits
            if week <= h.onset_week < week + weeks
        ]
        return rows, hits

    tally, clock, count = Tally(), {}, {}
    forecasts, costs = [], []  # rollouts: (first real week, the model's cost of each week); the real cost of each week

    def timed(name: str, start: float) -> None:
        clock[name] = clock.get(name, 0.0) + time.process_time() - start
        count[name] = count.get(name, 0) + 1

    def roll(first, arrays, hits) -> list:
        """The cost of each week of a rollout from the observation ``first``, the simulator on ``arrays``."""
        policy, w, seen_obs, spent = copy.deepcopy(player), m.window(first, rollout, arrays, hits), first, []
        for _ in range(rollout):
            spent.append(_spent(m.step(w, policy.act(seen_obs)).costs))
            seen_obs = m.flat(first, w)
        return spent

    done = False
    while not done:
        week = int(obs["week"][0])
        planner._remember(obs)
        start = time.process_time()
        seen = m.window(obs, 1, pending=pending)
        timed("window of 1 week", start)
        exact = m.window(obs, 1, *true_network(week, 1))

        mine, theirs = _state_dicts(seen.state, week - 1), _state_dicts(ep.state)
        tally.add("state: stock", seen.state.stock, ep.state.stock)
        for name in mine:
            tally.keyed(f"state: {name}", mine[name], theirs[name])
        tally.add("state: backlog", seen.backlog, ep.state.backlog)

        start = time.process_time()
        back = m.flat(obs, seen)
        timed("flat", start)
        _next(tally, "round trip: ", back, obs, model)  # no week played: last week's fields are the observation's own

        if rollout and (week - 1) % every == 0 and week + rollout - 1 <= T:
            start = time.process_time()
            policy = copy.deepcopy(player)
            timed("copy of the agent", start)
            start = time.process_time()
            ahead = m.window(obs, rollout, pending=pending)
            timed(f"window of {rollout} weeks", start)
            start = time.process_time()
            m.restart(ahead)
            timed("fresh state of a window", start)
            begin, seen_obs, spent = time.process_time(), obs, []
            for _ in range(rollout):
                start = time.process_time()
                action = policy.act(seen_obs)
                timed("rollout: the agent's act", start)
                start = time.process_time()
                spent.append(_spent(m.step(ahead, action).costs))
                timed("rollout: the simulator's step", start)
                start = time.process_time()
                seen_obs = m.flat(obs, ahead)
                timed("rollout: flat", start)
            timed(f"rollout of {rollout} weeks, all", begin)
            real_rows, hits = true_network(week, rollout)
            views = {"as observed": spent, "true network": roll(obs, real_rows, hits)}
            if by_field:
                guessed = m.forecast(obs, rollout, pending)
                for group, names in GROUPS.items():
                    names += tuple(now for now, of in model.L.NOW_FIELDS if of in names)
                    mixed = dict(guessed) | {name: real_rows[name] for name in names}
                    views[f"true {group}"] = roll(obs, mixed, hits if group == "fabs and plants" else ())
            forecasts.append((week, views))

        action = player.act(obs)
        start = time.process_time()
        guess = _record(m.step(seen, action), n_slots)
        timed("step", start)
        sure = _record(m.step(exact, action), n_slots)
        after = m.flat(obs, exact)
        obs, _reward, term, trunc, _info = env.step(action)
        done = term or trunc
        real = _record(ep.traj.records[-1], n_slots)
        costs.append(_spent(ep.traj.records[-1].costs))
        for name in RECORD:
            tally.add(f"week, true network: {name}", sure[name], real[name])
            tally.add(f"week, as observed: {name}", guess[name], real[name])
        if not done:
            _next(tally, "next observation: ", after, obs, model)

    costs = np.array(costs)
    rolled = [
        (week, {k: np.array(v) for k, v in views.items()}, costs[week - 1 : week - 1 + rollout])
        for week, views in forecasts
    ]
    return {"tally": tally, "clock": clock, "count": count, "rolled": rolled, "weeks": len(costs)}


def table(header: list[str], rows: list[list]) -> None:
    def cell(v) -> str:
        return (
            v
            if isinstance(v, str)
            else f"{v:,}"
            if isinstance(v, int)
            else f"{v:.2e}"
            if 0 < abs(v) < 1e-3
            else f"{v:,.4f}"
        )

    body = [[cell(v) for v in row] for row in rows]
    width = [max(len(x) for x in col) for col in zip(header, *body)]
    for row, raw in zip([header, *body], [header, *rows]):
        print("   " + "  ".join(x.ljust(w) if isinstance(v, str) else x.rjust(w) for x, w, v in zip(row, width, raw)))


def main(
    agent: str = "agents/anastasiia_rules_v3",
    task: str = "small",
    entropy: int = 111,
    episodes: int = 2,
    first: int = 0,
    rollout: int = 8,
    every: int = 4,
    pending: bool = False,
    by_field: bool = False,
    grids: str = "none",
    n_jobs: int = 2,
    out: str = "",
) -> None:
    """Print how far the model is from the environment (see the module docstring).

    Args:
        agent: the agent that plays, a folder or a name of agents/ (its copy plays the rollouts).
        task: small or full.
        entropy: the root of the episodes.
        episodes: how many episodes, from ``first``.
        first: the first episode.
        rollout: weeks of a rollout; 0: no rollouts.
        every: a rollout starts every this many weeks.
        pending: the model switches announced prohibitions on from their effective week.
        by_field: repeat every rollout with one group of the network's fields true (``GROUPS``).
        grids: the forecast of the grids' output in the network "as observed": none (a spell stays), blend or step
            (``grid_recovery.json``, see ``model.py``).
        n_jobs: worker processes.
        out: a JSON file for the numbers.

    """
    runs = Parallel(n_jobs=n_jobs)(
        delayed(episode)(agent, task, entropy, n, rollout, every, pending, by_field, grids)
        for n in range(first, first + episodes)
    )
    tally, clock, count = Tally(), {}, {}
    for r in runs:
        tally.merge(r["tally"])
        for name, value in r["clock"].items():
            clock[name] = clock.get(name, 0.0) + value
            count[name] = count.get(name, 0) + r["count"][name]
    weeks = sum(r["weeks"] for r in runs)
    print(f"{agent} on {task}, episodes {first}..{first + episodes - 1} of root {entropy}: {weeks} weeks\n")
    groups = (
        ("1. The state rebuilt from the observation", "state: "),
        ("2. Round trip: that state written back as an observation", "round trip: "),
        ("3a. One week ahead with the week's true network", "week, true network: "),
        ("3b. One week ahead with the network as observed", "week, as observed: "),
        ("4. The next observation (true network)", "next observation: "),
    )
    for title, prefix in groups:
        print(title)
        names = [n for n in tally.off if n.startswith(prefix)]
        table(["what", "share off", "weeks off", "weeks"], [[r[0][len(prefix) :], *r[1:]] for r in tally.rows(names)])
        print()

    rolled = [x for r in runs for x in r["rolled"]]
    summary = {}
    if rolled:
        real = np.array([r for _w, _v, r in rolled])  # (rollouts, weeks, [all, shed, unserved demand])
        whole = real[:, :, 0].sum()
        print(
            f"5. Rollouts of {rollout} weeks by a copy of the agent on the model: {len(rolled)}. The model's cost "
            "against"
        )
        print("   the environment's: of all the weeks (as shares of the environment's whole cost) and week by week")
        rows = []
        for title in rolled[0][1]:
            gap = np.array([v[title] for _w, v, _r in rolled]) - real
            row = [title]
            for part in range(3):
                run = gap[:, :, part].sum(1)
                row += [float(np.abs(run).sum() / whole), float(run.sum() / whole)]
            rows.append(row)
            summary[title] = {
                "share_off": row[1],
                "bias": row[2],
                "shed_off": row[3],
                "unserved_off": row[5],
                "n": len(rolled),
            }
        table(["simulator's network", "all: off", "bias", "shed: off", "bias", "unserved: off", "bias"], rows)
        print()
        rows = []
        for title in ("as observed", "true network"):
            gap = np.array([v[title] for _w, v, _r in rolled]) - real
            rows.append(
                [title, *(float(np.abs(gap[:, h, 0]).sum() / real[:, h, 0].sum()) for h in range(real.shape[1]))]
            )
        table(["share off by week", *(f"week {h + 1}" for h in range(real.shape[1]))], rows)
        print()
    print("6. CPU seconds on this machine, per call")
    table(["what", "seconds", "calls"], [[name, clock[name] / count[name], count[name]] for name in clock])
    if out:
        kept = {
            "agent": agent, "task": task, "entropy": entropy, "episodes": [first, episodes], "weeks": weeks,
            "share_off": {n: tally.off[n] / max(tally.total[n], 1e-300) for n in tally.off},
            "weeks_off": {n: tally.bad[n] / max(tally.weeks[n], 1) for n in tally.off},
            "seconds": {n: clock[n] / count[n] for n in clock},
            "rollout": summary,
        }  # fmt: skip
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
