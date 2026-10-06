# Brief: a better planner for ShockBench-Flow (shared by four parallel workers)

You work in your own copy of a repository (the path is in your task). Never touch any other folder, never run git
commands that write, never upload anything. Run Python only through `uv run` from the copy's root. Read `AGENTS.md`
there first: it describes the benchmark (a weekly supply-network control task; the score, RSS, is 0 for the naive rule
and 1 for the "clairvoyant plan").

## Where we are

- `agents/mpc/agent.py` is our best agent: each week it builds a linear program (LP) of the next weeks from the
  observation, solves it with SciPy and plays the first week. The LP builder is the benchmark package's own
  (`shockbench_flow.oracle.lp.build_lp`, called through `shockbench_flow.policies.lp_common.rolled_lp`); a copy of
  those modules sits in `agents/mpc/sbfv/`. It scores about 0.69 on our tuning episodes (network `small`, root 111).
- `team/experiments/mpc_foresight.py` is the test bench. It plays the package's `mpc_det` (the same planner) with part
  of its forecast replaced by the episode's true future. With everything known and a 52-week horizon it scores
  **0.7446** on 64 episodes (by harm level: 0.665, 0.818, 0.817, 0.821); with nothing known, 0.6826.
- So even a planner that knows the whole future loses a quarter of the scale. That is the planner's own fault, and it
  is what we want to fix. **The goal: raise the "everything known, horizon 52" score from 0.745 toward 0.85.**

## What we know about the loss (read this carefully)

- The "clairvoyant plan" (score 1) is the optimum of one LP over the episode, never played in the simulator. That LP
  leaves some of the simulator's rules free (`oracle/lp.py`, module docstring, "Planning rules" and "Remaining
  relaxation"): how a grid's electricity is split between base load and fabs (the simulator serves base load first at
  every `base_first` grid: `dynamics/production.py::allocate_energy`), how many lots a fab starts, how a packaging
  plant (OSAT) splits its throughput. An agent controls none of these: it only chooses flows on routes and tanker
  releases at straits.
- The MPC's window LP (`planning_rules=True`) adds those rules **for the first week of the window only**; later weeks
  stay relaxed. Measured with `team/experiments/plan_vs_exec.py` (everything known, horizon 52): each week's realised
  cost equals the plan's first-week cost component by component, but "cost so far + plan to go" rises every week,
  about 226 billion USD per episode in total (of about 990 billion between naive and clairvoyant). The plan keeps
  expecting a future the simulator will not allow.
- `team/experiments/rules_bound.py`: the full-horizon LP with `planning_rules=True` still scores 0.997, because the
  rules bind week 1 only. It is not an achievable bound.
- Tried and failed: pricing shed base load in every week of the window (the package prices it in week 1 only) made the
  real agent worse by 0.019. A likely reason: the price is far above the true cost of shed load, so over many weeks it
  distorts every trade-off; it is a priority trick that only works for one week.
- We do not know how much of the 0.25 is recoverable. Finding that out is part of the job.

Costs are dominated by two components: unserved chip demand (`shortage`) and unserved base load (`shed`). Chain:
fuel -> grid -> electricity -> fab (wafers + electricity -> raw chips) -> packaging plant -> market.

## How to measure

```
uv run python team/experiments/mpc_foresight.py --task=small --episodes=16 --horizon=52 --only=everything --n_jobs=2
```

- Episodes are 0..N-1 of root 111. A score on 16 episodes is not comparable with the 0.7446 on 64: always run the
  unchanged bench on the same N first and compare with that. Explore on 16 episodes, confirm a promising change on 64
  (`--episodes=64`). Use `--n_jobs=2` (three other workers share this 10-core machine); a 16-episode run at horizon 52
  takes about 2 minutes, a 64-episode run about 8.
- `--only="nothing (the MPC as it is)"` gives the same planner without foresight: report it too for your best
  variant, since the real agent has no foresight.
- Write your planner as a new script `team/experiments/planner_<your letter>.py` that reuses `mpc_foresight.py`
  (import it, subclass or copy its `episode` function) so the numbers are produced the same way. Do not edit files
  under `.venv/`; to change the LP, subclass, wrap or post-process the model (`LPModel` has `A_eq`, `b_eq`, `A_ub`,
  `b_ub`, `lb`, `ub`, `cost`, `salvage`, `meta`, `keys`, `index`, `objective()`), or copy the builder module into
  `team/experiments/` and edit the copy.
- Record the CPU seconds per week of your planner (mean and max, `time.process_time` around the weekly solve): the
  real budget is 2 s per week on `small`. A variant over budget is still worth reporting if it shows what is possible.

## What to report (your final message; it is all the lead sees)

1. What you built, in a few sentences, and the path of your script.
2. A table: variant, episodes, score with everything known (horizon 52), the unchanged bench on the same episodes,
   the difference; and for the best variant the score with nothing known and the CPU seconds per week.
3. What you learned about where the loss comes from, including what did not work.
4. What you would try next.

Report only what you ran. A negative result is useful; an invented or untested number is not. If you run out of
budget before finishing, say exactly what is done and what is not. Stop after about 15 evaluation runs or once a
change is confirmed on 64 episodes, whichever comes first.
