# Search lab: the simulator as the judge of a better trajectory (7 October)

Neither rules nor a linear program: the hybrid agent (`agents/anastasiia_hybrid_chiplp`) is frozen and used as the
starting point; a search changes the actions it played and keeps a change only when the simulator itself says the
episode got cheaper. The fast replay and the moves come from `../mpc_lab/planners/planner_LS.py`, which starts from a
plan of the mixed-integer program; here the start is what an agent really played.

## The question

How far is the hybrid from the best trajectory the simulator can execute, and in which decisions? The plan that knows
the future says 0.08 to 0.09 of score (`hub/FINDINGS.md`), but the simulator does not execute that plan: it burns gas
it holds, starts every wafer, releases straits by its own rule. A search judged by the simulator finds only what is
executable.

## Three steps, each a gate for the next

| step | what the search knows | time allowed | what it answers | go on if |
| --- | --- | --- | --- | --- |
| 1 | the whole episode (the true network of every week) | unlimited | the executable room near the hybrid's trajectory, split by fuel, wafers, chips | the gain is at least 0.02 |
| 2 | only what the agent has seen: the network "stays as it is" | unlimited | how much of step 1 survives without the future | at least half survives |
| 3 | the same | the week's CPU budget (2 s on Small, 4 s on Full) | a submittable agent: the plan carried from week to week, a few replays a week | it beats the hybrid in a paired comparison |

Step 1 is an upper bound for step 2, step 2 for step 3; each is also a lower bound of the true optimum (a local search
with a budget). Episodes: root 444 (the root of plan statistics; no agent is tuned on it), so the result sits beside
the plan with "base load first" on the same episodes.

## Step 1: `search.py`

```bash
uv run python lab/anastasiia/search_lab/search.py check agents/anastasiia_hybrid_chiplp --task=small --episode=0
uv run python lab/anastasiia/search_lab/search.py run agents/anastasiia_hybrid_chiplp --task=small --episodes=16 \
    --moves=fuel,fuelwafer --budget=80000 --sweeps=8 --n_jobs=8 --out=outputs/search_lab/small_444_x16_b80000.pkl
uv run python lab/anastasiia/search_lab/changes.py outputs/search_lab/small_444_x16_b80000.pkl --moves=fuel
```

- The agent plays the episode once; its weekly actions are kept in `outputs/search_lab/played/`.
- The replay (`dynamics.sim.step` with the state of every week kept) gives the environment's cost to the cent on Small
  and on Full (`check`). One full replay: 0.015 CPU s on Small, 0.14 s on Full.
- A move changes one dispatch (slot, week): scale by 0, 0.5, 1.5, 2, or move all or half of it one or two weeks earlier
  or later; a valve (terminal into grid) may also pass everything its terminal holds. First improvement; dearest
  dispatches first. `--moves` restricts the slots: `fuel` (or its parts `valve` and `order`), `wafer`, `chips` (raw and
  packaged), `fuelwafer`, `all`; each set is searched separately from the agent's own actions.
- `changes.py` prints what a search changed: requests by fuel and destination per third of the episode, and whether
  the raised dispatches had room on their first edge.

### Small, root 444, episodes 0 to 15 (the hybrid as played: 0.8444)

| moves | replays per episode | score | gain | bn USD per episode |
| --- | --- | --- | --- | --- |
| fuel, up to 3 passes | 26,000 | 0.8686 | +0.0243 | 23.8 |
| fuel, up to 8 passes | 65,000 | 0.8701 | +0.0257 | 25.2 |
| fuel and wafers together, up to 8 passes | 75,000 | 0.8720 | +0.0276 | 27.0 |
| valves only (terminal into grid) | 9,500 | 0.8507 | +0.0063 | 6.9 |
| orders only (from a source) | 16,500 | 0.8449 | +0.0005 | 0.6 |
| wafers only | 4,700 | 0.8472 | +0.0028 | 2.9 |
| chips after the fab only (8 episodes, 3,000 replays) | | | +0.0004 | 0.7 |

On the same episodes: v2 0.8314, the plan with "base load first" replayed blindly 0.8600, its own cost 0.9504.

- The room near the hybrid's trajectory is in fuel: +0.026, close to converged (doubling the budget added 0.0014). It
  puts the hybrid above the blind replay of the plan; 81 bn of the 105 bn to the plan's own cost stay out of reach of
  these moves.
- The gain needs orders and valves changed together: valves alone give +0.006, orders alone nothing. More than half of
  it (14.7 of 25.2 bn) is in the first third of the episode.
- What changes (`changes.py`): the valves pass more gas into every grid (requests up by 13 to 36 thousand units per
  episode and grid), crude into JP and EU too; orders from the sources fall (5 to 9 thousand units of gas per
  terminal), mostly in the last third. 97% of the units added went on edges that had room: nothing is taken from
  another grid.
- Wafers: the search drops dispatches (the hybrid sends too many); chips after the fab are left as the linear program
  set them.

### Full, root 444, episodes 0 to 7 (the hybrid as played: 0.8109)

| moves | replays per episode | score | gain | bn USD per episode |
| --- | --- | --- | --- | --- |
| fuel | 8,000 | 0.8140 | +0.0031 | 9.6 |
| wafers | 8,000 | 0.8138 | +0.0029 | 9.0 |
| fuel, an hour per episode (the valves may also pass everything) | 21,000 | 0.8191 | +0.0082 | 25.1 |

Floors only: on Full one pass over the fuel dispatches is about 70,000 replays at 0.14 s each, and the hour-long run
stopped on time, not on convergence. In it the valves give 14.6 bn and the orders 10.5; 15.9 of the 25.1 bn are in the
first third of the episode, as on Small.

## Step 2: `online.py`

```bash
uv run python lab/anastasiia/search_lab/online.py agents/anastasiia_hybrid_chiplp --task=small --episodes=16 \
    --moves=fuel --budget=700 --horizon=2 --straits=young --n_jobs=8 --oracle=outputs/search_lab/small_444_x16_b80000.pkl
```

Week by week in the true episode: the rest of the episode is modelled as the network of this week kept for every
later week; the plan (at first the actions the agent played) is searched in that model, this week's dispatches first;
week t of the plan is played for real. `--truth` gives the model the true network of every week (a check of the weekly
loop itself); `--freeze` freezes one part only; `--straits` says what a disrupted strait does next (`frozen`, `open`
next week, or `young`: a disruption seen for at most 3 weeks ends after one more). Every move the model keeps is also
valued in the true episode (the audit lines of the output).

| model of the future | episodes | score | against the hybrid |
| --- | --- | --- | --- |
| the true network (a check) | 16 | 0.8541 | +0.0097, better in 16 of 16 |
| frozen, everything stays | 16 | 0.8113 | -0.0330, worse in 14 of 16 |
| frozen, a young strait disruption ends | 16 | 0.8119 | -0.0325, worse in 14 of 16 |
| the same, only moves the model values above 1 bn | 16 | 0.8414 | -0.0030, better in 7, worse in 8 |

- Even with the true network the weekly loop keeps 38% of step 1's gain: a pass week by week, one dispatch at a time,
  cannot build the changes over several weeks that the search over the whole episode finds.
- With the frozen network the loop loses. The audit on the 16 episodes (straits "young"): the 469 moves an episode
  that the model values under 1 bn promise 19 bn and lose 41 bn in truth; the 12 it values above 1 bn promise 42 bn and
  really save 10. On single episodes the large moves lose too (episodes 2 and 3: 6 moves above 3 bn promise 89 bn and
  lose 20). Keeping only the moves above 1 bn does not turn it: 13 moves an episode are kept, half of them really
  better, and the score is 0.003 below the hybrid.
- Which part of "it stays as it is" misleads, episode 9 (the true network gives +0.0155): freezing the straits alone
  -0.0372; power alone +0.0111; edges +0.0088; prohibitions +0.0148; supply, fabs, demand +0.0155.
- The deeper cause: the plan of later weeks was recorded in the true episode, and in the frozen world the network goes
  another way, so the model values today's move against a future that is not the plan's. An honest version replays the
  agent itself in the model, not its recorded actions: about 78 s per replay of the hybrid on Full against 4 s a week.

**The gate is not passed: step 3 (a search inside the week's budget) is not built.**

## The same night, in parallel

A cloud session ran the same search from the hybrid's trajectory with its own code
(`../mpc_lab/planners/planner_LSF.py`, rows in `hub/tried/mpc.md`): +24.4 bn per episode on episodes 0 to 11, against
23.8 to 25.2 here on 0 to 15. Two independent runs: the fuel result stands as verified twice. Its further steps answer
what this lab would have tried next: shifting the fuel rules' constants towards the search's trajectory gives nothing,
and a weekly choice among fuel variants by replaying the agent itself in the simulator loses too
(`../mpc_lab/reports/lsf_signal.md`).

## What stays

Step 1 is a teacher the simulator can execute: +0.026 on Small in fuel, with the actions that get it. Whether its
corrections can be told from what the agent sees in the week (a rule or a small learned model on the valves and the
orders together) is the open question; the search inside the agent is not the way to it.
