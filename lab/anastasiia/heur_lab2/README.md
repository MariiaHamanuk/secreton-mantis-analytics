# Heuristics lab 2: what the simulator does by itself, used in the rule agent (6 October)

The second round of work on the rule agent (no linear program in `act`). The first round is `../heur_lab/` and gave
`agents/anastasiia_rules_fuelchip`. This round started from one fact of the MPC work (`hub/tried/mpc.md`): the
simulator generates power, starts lots, packages chips and releases containers by itself, and the agent only decides
what to ship. The result is `agents/anastasiia_rules_v2/`; the numbers are in `hub/tried/heuristics.md`.

## How the work was done

1. The base agent was compared with the plan that knows the future and obeys "base load first" (a mixed-integer
   program, `../stats_lab/plan_stats.py`) on the same 40 episodes of root 444, quantity by quantity: lots per fab, chips
   per market, shed per grid, fuel into each grid. `tools/account.py` does the agent's side from the simulator's own
   weekly records (lots started, raw chips disposed of, each fuel segment's output, the free segment's output).
2. The gap, 142 bn USD per episode, split into chips 66, shed 68 and other 7; the shed part split further into free
   energy thrown away (25), energy of lots whose chips are disposed of (11) and fuel that never arrived (31).
3. Three Sonnet subagents in git worktrees took one loss each (chips, fuels complete together, power nobody takes); the
   lead took the rest (nuclear fuel on Full, straits, buffers), merged and measured. The brief they worked from is
   `brief.md`.

## What is here

- `tools/account.py`: where an agent loses, with `--plan` beside the best plan (root 444) and `--task=full`.
- `brief.md`: the subagents' brief (rules of the lab, the simulator's weekly steps, the measured losses). Its paths point
  to `outputs/heur2/`, the local working folder of that day.
- `reports/throttle.md`, `reports/pull.md`: the two subagents' reports, every variant with its paired difference.
- `reports/sync_unmeasured.md`, `reports/sync_unmeasured.diff`: the third subagent's notes and its diff against the base
  `fuel_part.py`. Its session could not run the lab's tools, so nothing in them is measured and the code was never run;
  the rule for gating fuels was written and measured by the lead instead (`FuelRules.gate`).
- `reports/base_small_444.txt`, `reports/v2_small_444.txt`: `account.py --plan` for the base agent and for the result.

The subagents' worktrees were removed afterwards: paths in the reports that point into `.claude/worktrees/` or
`outputs/heur2/` are of that day only. Every variant in their tables is a switch of the final code (`PARAMS`, `FUEL`).

## Rules of thumb that came out of it

- A fab gets `clip(sliver - sum of the fuels' shortfalls, 0, what it asks)`. Any fuel whose weekly burn exceeds the
  sliver must come in whole weeks: lng everywhere, crude at JP and EU on Small and at five grids on Full.
- Every segment of a grid runs at one load factor. A grid that offers more than the load takes wastes free energy;
  fuel saved by offering less is worth something only at a grid that is short at other times.
- A buffer is worth more than a tidy stock: capping orders by storage, or holding lots back on the chain's inventory,
  lost every time. Disposal at a full terminal costs almost nothing; an empty one costs VOLL.
- What is cheap to hold and can be cut off must be brought early (nuclear fuel on Full).
- Measure on Full separately: the largest single gain of this round does not exist on Small.
