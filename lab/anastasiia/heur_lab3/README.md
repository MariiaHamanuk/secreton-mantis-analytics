# Heuristics lab 3: how much room the rule agent has left (6 October)

A review, not a new agent: where `agents/anastasiia_rules_v2` still loses, and whether more rules can add 0.05 to 0.1
of score. The first two rounds are `../heur_lab/` and `../heur_lab2/`. No agent code changed here; the numbers are in
`hub/tried/heuristics.md` (section "Третя хвиля") and `hub/FINDINGS.md`.

## How the work was done

Four Sonnet subagents, one question each, read-only on the repository, 2 worker processes each; the lead measured the
rest and checked their key numbers against their logs.

1. `reports/small_gap.md`: the gap to the plan that knows the future and obeys "base load first" (Small, root 444,
   40 episodes), split by period, by how much the network changes after week 1, by chips and by fuel.
2. `reports/full_anatomy.md`: the same accounting on Full beside the board's clairvoyant plan (root 111, 24 episodes
   with the LP, 64 for the agent's own records).
3. `reports/bestofk.md`: 19 one-switch variants of v2 and the best of them per episode chosen with hindsight, an upper
   bound on any selector of parameter variants.
4. `reports/full_ceiling.md`: the plan with "base load first" on Full (root 444, 8 episodes, 300 s per solve), its own
   cost, its blind replay and v2 on the same episodes.

Lead: the per-episode records of the formal models (`hub/eval/records/`), `tools/surprise.py` on 256 episodes of Small
and 64 of Full, six switches of the chip shipping rules.

## What is here

- `tools/costs.py`: cost per episode of several agent folders on the same episodes, each one's paired difference with
  the first, and the best of them per episode with hindsight. `--out` keeps the rows and skips folders already played.
- `tools/surprise.py`: a model's score against how much the true network departs from its week-1 state (the
  generator's marks, no agent played), by thirds and for the quietest fifth of the episodes.
- `tools/plan_gap.py`: an agent's weekly arrays (`../heur_lab2/tools/account.py --save`) beside the plan's
  (`../stats_lab/plan_stats.py`), on Small or Full: unserved demand by chip and shed by grid per quarter, lots and
  energy by fab, chips shipped out of fabs and plants, fuel moved into each grid while it can still be burned.
- `reports/`: the four subagents' reports as handed back. Their paths under `outputs/heur3/` are of that day only
  (local, not in git); the tools they call `tools/diag.py`, `balance.py`, `routes.py` and `pair.py` were copies of
  `hub/eval/diag.py`, `balance.py`, `routes.py` and `compare.py`.

## What came out of it

- Knowing the future is not what the rule agent lacks. After week 1 the network barely changes (median 0.3% of fuel
  capacity-weeks and 2% of chip capacity-weeks on Small), and the agent's score in the quietest fifth of the episodes
  is about the same as elsewhere (0.835 against 0.819 on Small, 0.850 against 0.825 on Full).
- The parameters of v2 are used up: the best of 19 variants per episode, chosen with hindsight, is +0.006 on Small and
  +0.008 on Full.
- The plan's own cost (0.95 on Small, 0.93 to 0.94 on Full) is not reached by replaying its orders (0.87 on both); v2
  is 0.03 below that replay on both networks. The plan is ahead in every episode, not in a few.
- What the plan does differently is joint, not local: it makes fewer chips and sells more (v2 throws away or strands
  3.4 M `chip_le` of 12.8 M made on Small, the plan 0.4 M), and in the weeks when gas can still be burned it lands
  5.9 thousand more of it at the grids (EU 3.8, KR 1.7) and 2.2 thousand more crude at JP. When the plan burns its gas
  is no teacher: it keeps gas at a grid unburned while that grid sheds, which the simulator does not allow. v2's stock
  at the grids at the end is what it moved there in the last week; the plan simply does not move it.
- On Full 139 of the 512 bn USD between v2 and the board's clairvoyant are lots the clairvoyant starts at CN and SEA
  while those grids shed: out of reach.
