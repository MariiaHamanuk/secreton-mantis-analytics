"""Would a rollout on the model pick the same change of this week's action as the true future would?

    uv run python lab/anastasiia/rollout_lab/candidates.py --task=small --episodes=16 --out=rows.json
    uv run python lab/anastasiia/rollout_lab/candidates.py --task=full --episodes=8 --every=8 --out=rows.json
    uv run python lab/anastasiia/rollout_lab/candidates.py --rows=rows.json          # the tables again, nothing played

An agent plays real episodes. Every ``every`` weeks its copy plays ``weeks`` weeks on ``model.Model`` several times:
with its own action in the first week (the base), and with that action changed (``CANDIDATES``: all fuel requests
of the week times 1.5 or 0.5, no wafers this week). A candidate's score is its cost minus the base's over the first
H weeks (``HORIZONS``), bn USD per decision; negative means the change pays. The truth is the score on the true
network of those weeks with the copy playing its own rules from the second week on, over the longest horizon
(``check.py`` shows that this model is the environment's own to about half a percent of cost).

What is compared with that truth:

- the network: ``true``, ``seen`` ("as observed"), ``grids`` (as observed, the grids' output forecast by
  ``grid_recovery.json``: ``--grid_rule``);
- the later weeks: ``closed`` (the copy plays its rules every week), ``open`` (the base rollout's actions are
  replayed after the changed first week: no call of the agent), ``mixed`` (rules for ``MIXED`` weeks, then replayed);
- the horizon, and for a short one an end value: what the stocks, the cargo on the way and the lots in process left
  after ``CUT`` weeks are worth, as a weight per commodity fitted on the even episodes (the cost difference of the
  later weeks regressed on the difference of those quantities) and used on the odd ones.

Printed: how each way of scoring ranks the changes (rank correlation with the truth) and what picking the best of the
base and the changes by it is worth on the truth. A decision's gain is counted on its own window and the agent then
goes on unchanged, so gains of different weeks do not add up: the numbers say whether a way of scoring ranks changes
correctly, not what an agent would score.
"""

import copy
import json
from pathlib import Path

import check
import fire
import numpy as np
from joblib import Parallel, delayed


HORIZONS = (8, 16, 24)
CUT = 8  # weeks after which the end value is taken
MIXED = 4  # weeks the copy plays its rules in a ``mixed`` rollout
CANDIDATES = {"fuel x1.5": (0, 1.5), "fuel x0.5": (0, 0.5), "no wafers": (1, 0.0)}  # (chip_part's slot kind, factor)
BN = 1e9


def episode(agent: str, task: str, entropy: int, n: int, weeks: int, every: int, grid_rule: str) -> list:
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
    plain = model.Model(config, planner)
    models = {"seen": plain, "true": plain}
    if (check.HERE / "grid_recovery.json").is_file():  # written by grid_spells.py
        table = json.loads((check.HERE / "grid_recovery.json").read_text())
        models["grids"] = model.Model(config, planner, table, grid_rule)
    inst = plain.inst
    truth = {name: np.asarray(getattr(ep.marks, name)) for name in model.L.WINDOW_FIELDS}
    kind = np.asarray(player.chips.kind)
    K = len(inst.commodities)
    sources = set(inst.supply_nodes)
    ours = np.array([sl.node not in sources for sl in inst.stock_slots])
    slot_k = np.array([sl.k for sl in inst.stock_slots])

    def assets(state) -> list:
        """What the state holds, per commodity: stock off the sources and cargo on the way; lots in fabs; in plants."""
        held, fabs, plants = np.zeros(K), np.zeros(K), np.zeros(K)
        np.add.at(held, slot_k[ours], np.asarray(state.stock)[ours])
        for s in state.pipeline:
            held[s.k] += s.qty
        for fi, book in state.fab_wip.items():
            fabs[inst.nodes[inst.fabs[fi]].fab.product] += sum(book.values())
        for book in state.osat_wip.values():
            for lots in book.values():
                for k, q in lots.items():
                    plants[k] += q
        return np.r_[held, fabs, plants].tolist()

    def roll(m, first, arrays, hits, change, replay=None, rules: int | None = None) -> dict:
        """A rollout, its first action changed by ``change``; ``replay``: the base's actions after ``rules`` weeks."""
        rules = weeks if replay is None else rules
        policy = copy.deepcopy(player) if rules else None
        w, seen, spent, played, left = m.window(first, weeks, arrays, hits), first, [], [], None
        for h in range(weeks):
            action = policy.act(seen) if h < rules else replay[h]
            if h == 0 and change is not None:
                flows = np.array(action["flows"], dtype=float)
                flows[kind == change[0]] *= change[1]
                action = dict(action) | {"flows": flows}
            played.append(action)
            spent.append(m.step(w, action).costs.total())
            if h + 1 == CUT:
                left = assets(w.state)
            if h + 1 < rules:
                seen = m.flat(first, w)
        return {"cost": spent, "actions": played, "left": left}

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
            row = {"episode": n, "week": week}
            for view, m in models.items():
                arrays, events = (real, hits) if view == "true" else (None, ())
                base = roll(m, obs, arrays, events, None)
                for name, change in CANDIDATES.items():
                    ways = {"closed": roll(m, obs, arrays, events, change)}
                    if view != "grids":
                        ways["open"] = roll(m, obs, arrays, events, change, base["actions"], 0)
                        ways["mixed"] = roll(m, obs, arrays, events, change, base["actions"], MIXED)
                    for way, r in ways.items():
                        row[f"{view}: {way}: {name}"] = (np.array(r["cost"]) - np.array(base["cost"])).tolist()
                    row[f"{view}: left: {name}"] = (np.array(ways["closed"]["left"]) - np.array(base["left"])).tolist()
            rows.append(row)
        obs, _reward, term, trunc, _info = env.step(player.act(obs))
        done = term or trunc
    return rows


def _rank(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation; nan when one of the two is constant."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")


def _line(label: str, guess: np.ndarray, real: np.ndarray) -> list:
    """How ``guess`` ranks the changes, and what picking the best of them and the base by it is worth on ``real``."""
    zero = np.zeros((len(real), 1))
    mine, best = np.c_[zero, real][np.arange(len(real)), np.c_[zero, guess].argmin(1)], np.c_[zero, real].min(1)
    corr = float(np.nanmean([_rank(guess[:, j], real[:, j]) for j in range(real.shape[1])]))
    return [label, corr, float(mine.mean()), float(best.mean()), float(np.mean(mine > 1e-6))]


def report(rows: list, title: str = "") -> None:
    """The tables of the module docstring from the rows of ``episode``."""
    names, weeks = list(CANDIDATES), len(rows[0][f"true: closed: {next(iter(CANDIDATES))}"])
    horizons = [h for h in HORIZONS if h <= weeks]
    longest = horizons[-1]
    header = ["network, later weeks, horizon", "rank corr.", "picked", "best", "picks a loss in"]

    def score(view: str, way: str, h: int, subset=None) -> np.ndarray:
        use = rows if subset is None else [rows[i] for i in subset]
        return np.array([[sum(r[f"{view}: {way}: {name}"][:h]) for name in names] for r in use]) / BN

    truth = score("true", "closed", longest)
    print(f"{title}{len(rows)} decisions; the truth: true network, rules every week, {longest} weeks\n")
    print("1. Each change against the base on the truth, bn USD per decision (negative: it pays)")
    check.table(
        ["change", "mean", "pays in"],
        [[name, float(truth[:, j].mean()), float(np.mean(truth[:, j] < -1e-6))] for j, name in enumerate(names)],
    )

    print("\n2. Ways of scoring a change: rank correlation with the truth (mean over the changes), and the best of the")
    print(
        "   base and the changes picked by it, valued on the truth (bn USD per decision; 'best': picked by the truth)"
    )
    table = []
    views = [v for v in ("true", "seen", "grids") if f"{v}: closed: {names[0]}" in rows[0]]
    for view in views:
        for way in ("closed",) if view == "grids" else ("closed", "mixed", "open"):
            for h in horizons:
                table.append(_line(f"{view}, {way}, {h} weeks", score(view, way, h), truth))
    check.table(header, table)

    # the end value: what is left after CUT weeks, a weight per commodity fitted on the even episodes
    fit = [i for i, r in enumerate(rows) if r["episode"] % 2 == 0]
    test = [i for i, r in enumerate(rows) if r["episode"] % 2 == 1]
    if not fit or not test or CUT >= longest:
        return

    def left(view: str, subset) -> np.ndarray:
        return np.array([[rows[i][f"{view}: left: {name}"] for name in names] for i in subset])

    later = (score("true", "closed", longest, fit) - score("true", "closed", CUT, fit)).ravel()
    X = left("true", fit).reshape(len(later), -1)
    keep = np.abs(X).sum(0) > 0
    scale = np.abs(X[:, keep]).mean(0)
    A = X[:, keep] / scale
    w = np.linalg.solve(A.T @ A + 1e-3 * len(later) * np.eye(A.shape[1]), A.T @ later) / scale
    print(f"\n3. A horizon of {CUT} weeks with an end value (fitted on the {len(fit)} decisions of the even episodes),")
    print(f"   on the {len(test)} decisions of the odd ones, valued on the truth")
    real = truth[test]
    table = [_line(f"true, closed, {CUT} weeks, no end value", score("true", "closed", CUT, test), real)]
    for view in views:
        end = (left(view, test).reshape(-1, len(keep))[:, keep] @ w).reshape(len(test), len(names))
        table.append(_line(f"{view}, closed, {CUT} weeks + end value", score(view, "closed", CUT, test) + end, real))
    for view in views:
        table.append(_line(f"{view}, closed, {longest} weeks", score(view, "closed", longest, test), real))
    check.table(header, table)
    K, parts = len(keep) // 3, ("held", "in fabs", "in plants")
    worth = [f"{parts[i // K]} k{i % K}: {-w[j] * BN:,.0f}" for j, i in enumerate(np.flatnonzero(keep))]
    print("   what a unit left is worth, USD (commodity by its index): " + "; ".join(worth))


def main(
    agent: str = "agents/anastasiia_rules_v3",
    task: str = "small",
    entropy: int = 111,
    episodes: int = 4,
    first: int = 0,
    weeks: int = 24,
    every: int = 4,
    grid_rule: str = "blend",
    n_jobs: int = 2,
    out: str = "",
    rows: str = "",
) -> None:
    """Play the rollouts and print how each way of scoring ranks the changes (see the module docstring).

    Args:
        agent: the agent that plays, a folder or a name of agents/ (it must keep ``chips.kind``, as the rule agents do).
        task: small or full.
        entropy: the root of the episodes.
        episodes: how many episodes, from ``first``.
        first: the first episode.
        weeks: weeks of a rollout (the longest horizon scored).
        every: a decision is examined every this many weeks.
        grid_rule: how the ``grids`` network reads the spells' table: blend or step.
        n_jobs: worker processes.
        out: a JSON file for the rows (the weekly cost differences of every decision, change and way).
        rows: a file written by ``out``: print its tables, play nothing.

    """
    if rows:
        report(json.loads(Path(rows).read_text()), f"{rows}: ")
        return
    runs = Parallel(n_jobs=n_jobs)(
        delayed(episode)(agent, task, entropy, n, weeks, every, grid_rule) for n in range(first, first + episodes)
    )
    kept = [row for run in runs for row in run]
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(kept))
    report(kept, f"{agent} on {task}, episodes {first}..{first + episodes - 1} of root {entropy}: ")


if __name__ == "__main__":
    fire.Fire(main)
