"""Local search over the actions an agent played, the simulator as the judge: what a better trajectory near it is worth.

    uv run python lab/anastasiia/search_lab/search.py check agents/anastasiia_hybrid_chiplp --task=small --episode=0
    uv run python lab/anastasiia/search_lab/search.py run agents/anastasiia_hybrid_chiplp --task=small --episodes=16 \
        --moves=fuel,wafer,chips,all --budget=6000 --n_jobs=8

The question: how far is an agent from the best trajectory the simulator can really execute, and in which decisions
(fuel, wafers, chips after the fab)? The plan that knows the future cannot say: the simulator does not execute its
plan (``hub/FINDINGS.md``). Here the judge is the simulator itself, so whatever the search finds is attainable.

Method (the fast replay and the moves are those of ``lab/anastasiia/mpc_lab/planners/planner_LS.py``, which starts
from a plan of the mixed-integer program; this starts from what an agent played):

1. The agent plays the episode once under gymnasium; the weekly actions it sent are kept
   (``outputs/search_lab/played/``).
2. The same actions are replayed with the raw simulator step (``dynamics.sim.step``), with the state at the end of
   every week kept, so a change in week t replays only weeks t..T. ``check`` asserts that the replay's cost equals the
   environment's to the cent.
3. Moves on one dispatch (slot, week): scale it by 0, 0.5, 1.5 or 2, or move all or half of it to the week before or
   after (one or two weeks); a valve (terminal into grid) may also pass everything its terminal holds, in a week the
   agent kept it closed too. A move is kept only if the played cost of the whole episode falls (first improvement).
   Dispatches are tried in the order of their customs value. ``--moves`` names which slots may move: ``fuel`` (lng,
   crude, nuclear fuel), or its two parts ``valve`` (terminal into grid) and ``order`` (from a source); ``wafer``;
   ``chips`` (raw and packaged chips); ``fuelwafer``; ``all``. Each set is searched separately from the agent's own
   actions, so the gains say where the room is.

The search knows the whole episode (the true network of every week): it is an upper bound for an agent that does not,
and a lower bound of the best executable trajectory (a local search with a budget, not an optimum).
"""

import copy
import pickle
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
PLAYED = ROOT / "outputs" / "search_lab" / "played"
LEVEL_WEIGHTS = (0.50, 0.30, 0.15, 0.05)
MOVES = (  # (name, target week offset or None for a scale, fraction moved or scale factor)
    ("x1.5", None, 1.5),
    ("drop", None, 0.0),
    ("shift-1", -1, 1.0),
    ("shift+1", 1, 1.0),
    ("half", None, 0.5),
    ("x2", None, 2.0),
    ("half-1", -1, 0.5),
    ("half+1", 1, 0.5),
    ("shift+2", 2, 1.0),
    ("shift-2", -2, 1.0),
)
SETS = {  # move set -> the kinds of slots that may change
    "fuel": ("valve", "order"),
    "valve": ("valve",),  # fuel from a terminal into its grid: it acts in the same week
    "order": ("order",),  # fuel from a source: it arrives weeks later
    "wafer": ("wafer",),
    "chips": ("raw", "pack"),
    "fuelwafer": ("valve", "order", "wafer"),
    "all": ("valve", "order", "wafer", "raw", "pack"),
}


def world(task: str, entropy: int, n: int):
    """(instance, marks) of an episode: the true network of every week."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks

    inst, params = task_generator(task)
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train"))
    return inst.at_digest(marks.instance_digest), marks


def played(agent: str, task: str, entropy: int, n: int) -> dict:
    """The weekly actions the agent sent in one episode and the environment's cost in cents (kept on disk)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    folder = resolve(agent).resolve()
    path = PLAYED / f"{folder.name}_{task}_{entropy}_{n}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(str(folder))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    core, sent = u.core, []
    step = core.step

    def recording(action):
        sent.append(copy.deepcopy(action))
        return step(action)

    core.step = recording  # the environment's own conversion of the agent's arrays is what is recorded
    done = False
    while not done:
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    out = {"actions": sent, "J": int(core._ep.traj.J_cents)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


class Replay:
    """An episode played from weekly actions with the raw simulator step, the state of every week kept."""

    def __init__(self, inst, marks, actions):
        from shockbench_flow.dynamics.env import validate_action
        from shockbench_flow.dynamics.sim import initial_state

        self.inst, self.marks, self.T = inst, marks, inst.T
        self.flows, self.ov, self.holds = [], [], []
        for t, a in enumerate(actions, start=1):
            fl, ov, ho, _dropped = validate_action(inst, marks, t, a)  # masked entries are dropped, as the Env does
            self.flows.append(dict(fl)), self.ov.append(ov), self.holds.append(ho)
        s0 = initial_state(inst)
        s0.last = None
        self.states = [s0] + [None] * self.T  # states[t]: the end of week t
        self.costs = [0] * (self.T + 1)
        self.calls = 0
        self.t0 = 1  # the first week this replay plays
        self.J = self._run(1, self.flows, keep=True)

    @classmethod
    def from_week(cls, inst, marks, flows: list, ov: list, holds: list, state, t0: int):
        """A replay of weeks t0..T from ``state`` (the end of week t0 - 1); its cost counts those weeks only."""
        R = cls.__new__(cls)
        R.inst, R.marks, R.T = inst, marks, inst.T
        R.flows, R.ov, R.holds = flows, ov, holds
        R.states = [None] * (R.T + 1)
        R.states[t0 - 1] = state
        R.costs = [0] * (R.T + 1)
        R.calls, R.t0 = 0, t0
        R.J = R._run(t0, flows, keep=True)
        return R

    def _run(self, t0: int, flows: list, keep: bool) -> int:
        """Weeks t0..T from the state kept at the end of week t0 - 1; the episode's cost in cents."""
        from shockbench_flow.dynamics.sim import step, terminal_salvage
        from shockbench_flow.dynamics.state import cents

        st = copy.deepcopy(self.states[t0 - 1])
        J = sum(self.costs[1:t0])
        self.calls += 1
        for t in range(t0, self.T + 1):
            rec = step(self.inst, self.marks, st, flows[t - 1], self.ov[t - 1], self.holds[t - 1])
            J += rec.cost_cents
            if keep:
                self.costs[t] = rec.cost_cents
                st.last = None
                if t < self.T:
                    self.states[t] = copy.deepcopy(st)
        return J - cents(terminal_salvage(self.inst, st))

    def valid(self, t: int, s: int) -> bool:
        """The slot's own first edge is not prohibited for its commodity in week t (else the entry is dropped)."""
        e, k, _lane = self.inst.action_slots[s]
        return not self.marks.prohibited[t - 1][e, k]

    def trial(self, changes: dict):
        """(cost, flows) of the incumbent with ``changes`` {(week, slot): qty}; nothing is kept."""
        flows = list(self.flows)
        for (t, s), q in changes.items():
            if flows[t - 1] is self.flows[t - 1]:
                flows[t - 1] = dict(self.flows[t - 1])
            flows[t - 1][s] = q
        return self._run(min(t for t, _ in changes), flows, keep=False), flows

    def accept(self, changes: dict, flows: list, J: int) -> None:
        self.flows = flows
        self.J = self._run(min(t for t, _ in changes), flows, keep=True)
        assert self.J == J, (self.J, J)


def slot_kinds(inst) -> list[str]:
    """Per action slot: valve or order (a fuel), wafer, pack (a chip a market buys) or raw (the rest).

    A fuel slot is a valve when it is one edge from a node that is itself supplied by edges (a terminal) into a grid;
    every other fuel slot is an order.
    """
    fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
    wafers = {inst.nodes[f].fab.input for f in inst.fabs}
    sold = {d.k for d in inst.demands}
    fed = {e.head for e in inst.edges}
    out = []
    for e, k, lane in inst.action_slots:
        edge = inst.edges[e]
        if k in fuels:
            valve = lane is None and inst.nodes[edge.head].grid is not None and edge.tail in fed
            out.append("valve" if valve else "order")
        else:
            out.append("wafer" if k in wafers else "pack" if k in sold else "raw")
    return out


def held(R: Replay, t: int, s: int) -> float:
    """What the slot's origin holds of its commodity at the start of week t: all a valve can pass on this week."""
    e, k, _lane = R.inst.action_slots[s]
    return float(R.states[t - 1].stock[R.inst.slot_index[(R.inst.edges[e].tail, k)]])


def moves_of(R: Replay, t: int, s: int, valve: bool = False):
    q = R.flows[t - 1].get(s, 0.0)
    if valve:  # a valve may also pass everything its terminal holds, in a week it kept closed too
        everything = held(R, t, s)
        if everything > q + 1e-9:
            yield "all", {(t, s): everything}
    if q <= 0:
        return
    for name, dt, f in MOVES:
        if dt is None:
            yield name, {(t, s): q * f}
        elif R.t0 <= t + dt <= R.T and R.valid(t + dt, s):
            yield name, {(t, s): q * (1.0 - f), (t + dt, s): R.flows[t + dt - 1].get(s, 0.0) + q * f}


def search(
    R: Replay,
    kinds: list[str],
    allowed: tuple,
    budget: int,
    secs: float,
    sweeps: int = 3,
    last: int | None = None,
    by_week: bool = False,
    min_gain: int = 0,
    on_accept=None,
) -> dict:
    """First-improvement local search on the dispatches of the allowed kinds, dearest dispatches first.

    Only the dispatches of weeks ``R.t0 .. last`` are tried (``last`` None: to the end of the episode); ``by_week``
    tries them week by week, the earliest week first, the dearest first inside a week. A move is kept when it lowers
    the cost by more than ``min_gain`` cents; ``on_accept(change, gain)`` is called for every move kept.
    """
    v = np.array([R.inst.commodities[k].v for _e, k, _l in R.inst.action_slots])
    start, J0, calls0 = time.time(), R.J, R.calls
    by_move, by_kind, by_third = {}, {}, [0, 0, 0]

    def spent():
        return R.calls - calls0 >= budget or time.time() - start > secs

    for _sweep in range(sweeps):
        improved = False
        valves = [s for s, kind in enumerate(kinds) if kind == "valve" and kind in allowed]
        todo = []
        for t in range(R.t0, (last or R.T) + 1):
            row = R.flows[t - 1]
            tried = {s for s, q in row.items() if kinds[s] in allowed and (q > 1e-9 or kinds[s] == "valve")}
            tried |= {s for s in valves if R.valid(t, s)}  # a closed valve is not in the action: it may still open
            todo += [
                ((t if by_week else 0), -max(row.get(s, 0.0), held(R, t, s) if s in valves else 0.0) * v[s], t, s)
                for s in tried
            ]
        todo.sort()
        for _week, _value, t, s in todo:
            if spent():
                break
            for _again in range(3):  # after an improvement the same dispatch is tried again
                hit = False
                for name, change in moves_of(R, t, s, kinds[s] == "valve"):
                    if spent():
                        break
                    J, flows = R.trial(change)
                    tried = by_move.setdefault(name, [0, 0, 0])
                    tried[0] += 1
                    if J < R.J - min_gain:
                        gain = R.J - J
                        R.accept(change, flows, J)
                        if on_accept is not None:
                            on_accept(change, gain)
                        tried[1] += 1
                        tried[2] += gain
                        by_kind[kinds[s]] = by_kind.get(kinds[s], 0) + gain
                        by_third[min(2, 3 * (t - 1) // R.T)] += gain
                        improved = hit = True
                        break
                if not hit:
                    break
        if not improved or spent():
            break
    return {"before": J0, "after": R.J, "replays": R.calls - calls0, "seconds": time.time() - start,
            "by_move": by_move, "by_kind": by_kind, "by_third": by_third}  # fmt: skip


def one(agent: str, task: str, entropy: int, n: int, moves: tuple, budget: int, secs: float, sweeps: int) -> dict:
    p = played(agent, task, entropy, n)
    inst, marks = world(task, entropy, n)
    kinds = slot_kinds(inst)
    out = {"episode": n, "J": p["J"], "sets": {}}
    for name in moves:
        R = Replay(inst, marks, p["actions"])  # every set starts from the agent's own actions
        assert R.J == p["J"], (R.J, p["J"])
        out["flows"] = [dict(f) for f in R.flows]  # as played: {slot: quantity} per week
        out["sets"][name] = search(R, kinds, SETS[name], budget, secs, sweeps)
        out["sets"][name]["flows"] = R.flows  # after the search
    return out


def pooled(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    present = [s for s in (1, 2, 3, 4) if (level == s).any()]
    w = {s: LEVEL_WEIGHTS[s - 1] for s in present}
    return float(
        sum(w[s] * saved[level == s].mean() for s in present) / sum(w[s] * room[level == s].mean() for s in present)
    )


def check(agent: str, task: str = "small", entropy: int = 444, episode: int = 0) -> None:
    """The replay reproduces the environment's cost to the cent; how long one replay takes."""
    p = played(agent, task, entropy, episode)
    inst, marks = world(task, entropy, episode)
    R = Replay(inst, marks, p["actions"])
    print(f"{task} root {entropy} episode {episode}: environment {p['J']}, replay {R.J}, equal {R.J == p['J']}")
    kinds = slot_kinds(inst)
    print("slots by kind:", {k: kinds.count(k) for k in sorted(set(kinds))})
    t0 = time.process_time()
    for _ in range(5):
        R._run(1, R.flows, keep=False)
    print(f"a full replay: {(time.process_time() - t0) / 5:.3f} CPU s")


def run(
    agent: str,
    task: str = "small",
    entropy: int = 444,
    episodes: int = 8,
    first: int = 0,
    moves: str = "fuel,wafer,chips,all",
    budget: int = 6000,
    secs: float = 3600.0,
    sweeps: int = 3,
    n_jobs: int = 4,
    out: str = "",
) -> None:
    """Search each move set on ``episodes`` episodes and print the score before and after, and where the gain is.

    Args:
        agent: an agent folder or a name of agents/.
        task: small or full.
        entropy: the scenarios' root (444: the root of plan statistics, no agent is tuned on it).
        episodes: how many episodes, starting at ``first``.
        first: the first episode index.
        moves: which move sets to search, of fuel, wafer, chips, all.
        budget: replays per episode and move set.
        secs: seconds per episode and move set, whichever comes first.
        sweeps: passes over the dispatches at most (a pass without an improvement ends the search).
        n_jobs: workers (one episode each).
        out: a path to keep the results in (.pkl), optional.

    """
    from sbf_starter import scoring

    names = tuple(m for m in (moves if isinstance(moves, tuple | list) else str(moves).split(",")) if m)
    refs = list(scoring.episode_set(task, first + episodes, entropy=entropy, n_jobs=n_jobs, verbose=False).references)
    refs = refs[first : first + episodes]
    res = Parallel(n_jobs=n_jobs)(
        delayed(one)(agent, task, entropy, n, names, budget, secs, sweeps) for n in range(first, first + episodes)
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(pickle.dumps({"agent": agent, "task": task, "entropy": entropy, "results": res}))
    level = np.array([r["stratum"] for r in refs])
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    room = naive - np.array([r["J_clairvoyant_cents"] for r in refs], dtype=float)
    base = np.array([r["J"] for r in res], dtype=float)
    bn = 1e11  # cents per bn USD
    counts = [int((level == s).sum()) for s in (1, 2, 3, 4)]
    print(f"{agent} on {task}, root {entropy}, episodes {first}..{first + episodes - 1}; levels {counts}")
    print(f"budget: {budget} replays per episode and move set")
    print(f"as played: score {pooled(level, naive - base, room):.4f}\n")
    head = f"{'moves':>6} {'score':>7} {'gain':>8} {'bn/episode':>11} {'replays':>8} {'min':>5}"
    print(f"{head}   gain by third of the episode, bn")
    for name in names:
        after = np.array([r["sets"][name]["after"] for r in res], dtype=float)
        thirds = np.array([r["sets"][name]["by_third"] for r in res], dtype=float).mean(axis=0) / bn
        score, score0 = pooled(level, naive - after, room), pooled(level, naive - base, room)
        replays = np.mean([r["sets"][name]["replays"] for r in res])
        minutes = np.mean([r["sets"][name]["seconds"] for r in res]) / 60
        print(
            f"{name:>6} {score:7.4f} {score - score0:+8.4f} {(base - after).mean() / bn:11.1f} {replays:8.0f}"
            f" {minutes:5.1f}   {thirds[0]:.1f} / {thirds[1]:.1f} / {thirds[2]:.1f}"
        )
    for name in names:
        kinds, moves_ = {}, {}
        for r in res:
            for k, g in r["sets"][name]["by_kind"].items():
                kinds[k] = kinds.get(k, 0) + g
            for k, (tried, kept, g) in r["sets"][name]["by_move"].items():
                a = moves_.setdefault(k, [0, 0, 0])
                a[0], a[1], a[2] = a[0] + tried, a[1] + kept, a[2] + g
        print(
            f"\n{name}: gain by kind, bn per episode: "
            + ", ".join(f"{k} {g / bn / len(res):.1f}" for k, g in sorted(kinds.items()))
        )
        ranked = sorted(moves_.items(), key=lambda x: -x[1][2])
        cells = (f"{k} {kept}/{tried} ({g / bn / len(res):.1f})" for k, (tried, kept, g) in ranked)
        print("   moves kept / tried (bn per episode): " + ", ".join(cells))
    print("\nper episode (level, score as played, then after each move set):")
    for i, r in enumerate(res):
        cells = " ".join(f"{(naive[i] - r['sets'][m]['after']) / room[i]:.3f}" for m in names)
        print(f"  {r['episode']:>3}  {level[i]}  {(naive[i] - base[i]) / room[i]:.3f}  {cells}")


if __name__ == "__main__":
    fire.Fire({"check": check, "run": run})
