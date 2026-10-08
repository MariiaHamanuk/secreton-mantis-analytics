# Regime lab: the episode as a linear program inside the simulator's regimes (7 October)

The question of the day: is the planner (MPC) weak because it is given the wrong problem? The package's program
leaves the simulator's automatic steps free, a mixed-integer program that writes them exactly does not solve in time
(`hub/tried/mpc.md`). Here they are written exactly and without one integer variable.

## The idea

Every automatic step of the simulator is "the smaller of two limits":

| step | the limits |
| --- | --- |
| a fuel segment's burn | its capacity, its ration (last week's stock), the fuel on hand |
| a grid | base load first; the fabs get what is left, up to what they ask |
| a fab's lot starts | its capacity, the wafers on hand |
| a plant's packaging | its throughput, its raw chips |
| a source's lift, disposal, a strait's release | the availability and the storage; the storage; the queue and the throughput |

Once it is known which limit binds in every (week, element), a **regime**, each step is a linear equation and the
whole episode is a linear program: the package's oracle program plus rows and bounds that state the regimes (a
**cell**). The regimes are read from a trajectory the simulator played, so that trajectory lies in its own cell and
the cell's optimum can only be cheaper. The loop (`core.descend`):

1. play a plan in the simulator, read its regimes;
2. solve the cell (HiGHS, 0.3 s for the 52 weeks of Small);
3. play the solution; repeat while the played cost falls.

Where the solution sits on the border of two cells (both limits bind), the duals of the regime's constraint say
which side it pushes to, and the next pass reads the tie that way (`Episode.hints`): the loop walks from cell to
cell without a search. Every cost reported is the simulator's or the environment's, never the program's claim; the two
agree to 0.0–1 bn USD an episode.

## What is here

- `core.py`: `Episode` (a scenario or an agent's window: `simulate`, `regimes`, `cell`, `solve`, `hints`,
  `actions`) and `descend`. It runs on the package or on the copy an agent ships (`core.PKG = "sbfv"`) and solves
  with SciPy's bundled HiGHS (`scipy.optimize._highspy._core`, a private module: the same optimum and duals as
  highspy, and a warm re-solve after a bound change in 5 ms on Small).
- `plan.py starts`: the loop with the whole future known, from several starts.
- `closed.py run`: the loop week by week in the real environment (`Env`), the plan carried from week to week; the
  forecast is the package's persistence with chosen fields replaced by the scenario's own (`--known`, `--look`).
- `agents/regime_x/agent.py`: the same as an agent under gymnasium: the hybrid's rules play the weeks left on the
  simulator's model (`../rollout_lab/model.py`), that plan or last week's is improved by the loop, its first week is
  sent; the options that hedge the plan with the rules (`floor`, `lot_bonus`, `anchor`, `typed`).
- `agents/anastasiia_plan_regime/agent.py`: the merged agent (below), with a clock.
- `build.py`: assembles an agent's folder that stands alone in `outputs/regime_lab/agents/<name>/` (the hybrid's
  parts, `sim_model.py`, `plan_core.py`, the agent, `regime.json` for its numbers).
- `play.py`: plays agent folders under gymnasium and keeps every episode's cost (`outputs/regime_lab/play/`), so
  variants are compared in pairs without replaying the base.
- `teacher.py`: full-knowledge plans as (observation, action) files for imitation, in `../mpc_lab/teacher/`'s
  layout (`outputs/teacher_regime/`): Small, root 555, episodes 0 to 15: 0.840 to 0.925.

```bash
uv run python lab/anastasiia/regime_lab/plan.py starts --episodes=8 --which=hybrid,basefirst --n_jobs=3
uv run python lab/anastasiia/regime_lab/closed.py run --episodes=8 --known=everything --start=offline --passes=2
uv run python lab/anastasiia/regime_lab/closed.py run --episodes=8 --known=nothing --start=basefirst --passes=3
uv run python lab/anastasiia/regime_lab/build.py --name=regime_x
uv run python hub/eval/compare.py --base=agents/anastasiia_hybrid_hub --entropy=444 --episodes=8 outputs/regime_lab/agents/regime_x
```

## Results

Small, root 444, episodes 0 to 7; checked by one run unless said. Two readings of the score on this set (it has no
episode of harm level 3): as `sbf evaluate` scores it, and in brackets with the levels present weighing the same, the
reading of the numbers of 6 and 7 October in `hub/tried/mpc.md`.

**With the whole future known** (the plan as the simulator plays it; `plan.py starts`):

| | score | time |
| --- | --- | --- |
| `anastasiia_hybrid_chiplp` as played (the start) | 0.829 [0.852] | |
| the mixed-integer plan with exact rules, played (`hub/tried/mpc.md`) | [0.896] | 8 to 16 minutes an episode |
| the loop from the hybrid's trajectory, ties read the default way | [0.915] | 2 to 8 s an episode |
| the same with the duals' hints | 0.908 [0.920] | 2 to 13 s |
| the loop from the plan with base load first (a 60 s mixed-integer program) | 0.910 [0.917] | |
| the better of the two per episode | 0.912 [0.923] | |
| the plan with base load first, its own cost (an upper bound of anything executable; not proven in 4 episodes) | [0.95] | |

- One solve from the hybrid's trajectory is worth 32 bn USD on episode 0, more than the search of 65,000 replays
  found there (`../search_lab/`), and the simulator plays it to within 0.002 bn of the program's claim.
- Per episode the loop ends 0 to 15 bn above the bound in four episodes (0, 2, 4, 5) and 27 to 95 bn above it in the
  other four (1, 3, 6, 7): there the simulator's other rules cost little, and what is left is the choice of the
  weeks a grid runs its fabs, which the loop changes only at a border.
- The start decides the cell: from the oracle's plan the loop ends at [0.806].
- The same plans sent through the real environment week by week (`closed.py`, the plan carried, two passes a week)
  cost what the offline replay says: [0.920]; 0.15 to 0.27 CPU s a week, 0.8 s at most (this machine, busy).
- The neighbouring `../plan_lab/` writes the same rows on its own code and reports 0.899 with the future known on
  these episodes: the result stands on two implementations.
- Full, episode 0 of root 444, 8 passes: 2,378.7 to 2,115.8 bn (0.894 to 0.981 for the episode), not converged;
  about 30 s a pass on a busy machine: as it is, far over 4 s a week.

**Without the future** (`closed.py`, the persistence forecast, the plan carried, 3 passes a week, the first week's
plan from the base-load-first program on the forecast; scores in brackets' reading):

| what of the future is known | score |
| --- | --- |
| nothing | 0.842 |
| plants and supply | 0.842 |
| straits | 0.849 |
| grids' output | 0.850 |
| prohibitions | 0.859 |
| edge capacities | 0.890 |
| everything | 0.922 |

- Not knowing the future costs this planner 0.08, more than the 0.03 to 0.06 measured on the package's program: a
  tight plan leans on capacities. Edge capacities are most of it. They change in steps (a group of edges between two
  regions falls to a quarter or comes back, once or twice an episode), not as noise.
- "Cuts end with a chance per week" (the forecast's capacities drift back to nominal: `--recover`) scores 0.809:
  better where a cut did end (episodes 2 and 7), worse where it stayed.
- Episode 7 shows what the tight plan loses: its strait is cut for 24 weeks, the plan sees no way out for the chips and
  starts half the lots the hybrid does; the strait reopens and the hybrid sells what it kept making (150 bn).

**As an agent under gymnasium** (`agents/regime_x`: the rules' plan every week, two passes; no CPU limit):

| set | `anastasiia_hybrid_hub` | `regime_x` | paired difference (90% interval) |
| --- | --- | --- | --- |
| Small 444, episodes 0 to 7 | 0.838 | 0.857 | +0.018 (+0.002 to +0.037) |

CPU a week on this machine while it was busy: median 0.5 s, 1.2 s at the 95th percentile, 4 s in the first week (six
passes).

## Without the future: what makes the plan pay

A plan that decides everything on the forecast "it stays as it is" equals the hybrid (above). It is better in two
episodes of three and loses 0.06 to 0.13 in one of six: it trims what the rules keep as insurance, surplus lots and
fuel on the way, and the network then changes (root 111, episode 21: about eight weeks of the KR fab lost, 2.4 M
lots, 103 bn of unserved demand). Two ways of giving the insurance back, both confirmed on the tuning set.

**In the first lab agent** (`agents/regime_x`: the rules' plan every week, the whole horizon, two passes; Small,
root 111; paired with `anastasiia_hybrid_hub`, the interval a bootstrap over episodes, `play.py`):

| variant | episodes 0 to 23 | all 64 |
| --- | --- | --- |
| as it is | +0.002 (−0.012 to +0.015) | −0.003 (−0.014 to +0.007), `compare.py` |
| at least the rules' wafers (`floor`) | −0.000 | |
| at least the rules' wafers and fuel orders | +0.009 (−0.007 to +0.022) | |
| at least the rules' fuel at the terminals (`stock_floor`) | −0.002; with the orders +0.011 | |
| a bonus per lot started (`lot_bonus` 0.25) | +0.002 | |
| the bonus and at least the rules' fuel orders | +0.025 (+0.018 to +0.033) | +0.023 (+0.017 to +0.028), better in 54 |
| **a price for leaving the rules' plan** (`anchor`: 200,000 USD a unit of fuel ordered, 5,000 a wafer) | +0.026 (+0.017 to +0.035) | **+0.027 (+0.022 to +0.032), better in 57** |
| cut edges come back by the cut's type and age (`typed`) | +0.001; with the bonus and the orders +0.023 | |
| `lean_plan` of `../plan_lab/` (its costs) | +0.015 (+0.005 to +0.026) | +0.014 (+0.008 to +0.020) |

Paired with `lean_plan` on the 64 episodes the anchored variant is +0.013 (+0.008 to +0.019), better in 49.

**The merged agent** (`agents/anastasiia_plan_regime`, built by `build.py` into a folder that stands alone:
`outputs/regime_lab/agents/anastasiia_plan_regime/`). From `../plan_lab/`: the fuel rules keep the orders, a rollout
needs the fuel rules only, a window of 26 weeks ends with a value for the fuel and the chips left in the system. From
this lab: the cell program, the hints, a simplex started from last week's basis, grid-weeks switched on, the price for
leaving the rules' plan, a clock (`share` of the week's CPU budget). Small, root 111, episodes 0 to 23, no clock, CPU
seconds a week on this machine while it was busy (median / 95th percentile):

| variant (`regime.json`) | against the hybrid | CPU |
| --- | --- | --- |
| the default: `lean_plan`'s recipe on this lab's core | +0.017 (+0.006 to +0.028), better in 19 | 0.12 / 0.24 |
| two passes a week | +0.019 (+0.009 to +0.030) | 0.17 / 0.31 |
| three grid-weeks a week tried for a switch on (`switch` 3) | +0.027 (+0.015 to +0.038), better in 21 | 0.35 / 0.74 |
| a bonus per lot | +0.014 | 0.11 / 0.23 |
| the program may order more than the rules (`orders` "floor") | +0.005 (−0.010 to +0.019) | 0.11 / 0.24 |
| the window to the end of the episode | +0.016 | 0.14 / 0.42 |
| the price for leaving the rules' plan, the program decides the orders (`orders` "plan", `anchor`), the whole horizon | +0.025 (+0.016 to +0.034) | 0.26 / 0.64 |
| **the same in the window of 26 weeks** | **+0.031 (+0.020 to +0.043), better in 21** | 0.25 / 0.37 |
| the same, one pass | +0.025 | 0.19 / 0.28 |
| the rules' orders and the price on wafers only | +0.022 | 0.20 / 0.29 |

- The default reproduces `lean_plan` (+0.015 on these episodes).
- Switching a shedding grid-week on pays without the future too (+0.009 on the default): the plan no longer only
  inherits "this grid sheds this week".
- With the end value for chips a bonus per lot is one insurance too many; and the orders must not exceed the rules'
  freely ("floor"), only at a price.

## Small against Full: the ladder and the worth of a whole week (7 October, evening)

The plan gained on Small and lost on Full. The question was put the other way round: which part of the mechanic is
unfinished, so that Small hides it and Full shows it. Root 444, episodes 0 to 7 of each network, the same agent with
the same settings on both, one pass a week, no clock; differences are paired with the hybrid's record of the same
run. The scores and intervals were recomputed independently by `plan_lab/scores.py` from these records.

Tools added here:

| file | what it does |
| --- | --- |
| `where.py` | two recorded agents by cost item, by grid (shed load), by fab (lots) and by part of the episode |
| `play.py arrays` | a tag's weeks as the `.npz` that `plan_lab/ledger.py` reads: generation, energy to the fabs, sales |
| `probe.py` | the first weeks of one episode under an agent: one grid's shed, burn, stocks, lots and the agent's notes |
| `plan.py starts --hull=N` | the full-knowledge plan with rounds of whole weeks asked for by the hull |

**What differs between the networks under the hybrid.** Small is a deep energy deficit, Full sits at the margin.
Grid-weeks of grids with fabs (Small root 111 x 64, Full root 444 x 8):

| | Small | Full |
| --- | --- | --- |
| the fabs' worth that is started | 25% of capacity | 54% |
| grid-weeks shedding over 10% of the base load | 38% (94% of all shed load) | 13% (59%) |
| grid-weeks shedding under 2% of the base load, the fab idle | 4% | 9% (CN: 43%) |
| a fab's load as a share of its grid's base load | 0.4% to 7.6% | 0.2% (CN) to 9.9% |

**Where the plan's difference to the hybrid sits** (`where.py`). Shed load: +0.025 to +0.03 of the room on both
networks, in 63 episodes of 64 on Small and 8 of 8 on Full. On Small all of it comes in weeks 6 to 12, the first
crisis (31 bn USD an episode there against 25 over the whole episode: the plan gives some back later). Chips: about zero on Small, of unstable
sign; on Full minus 115 bn an episode in 0 episodes of 8, four fifths of it in weeks 53 to 91, and still a loss with
the true future inside the window. One fab explains most of it: CN's lots fall from a third of capacity to a tenth
from week 27 on, with or without the true future, while the full-knowledge plan keeps them.

**The ladder** (against the hybrid, 0.838 on Small and 0.836 on Full):

| variant | Small | Full |
| --- | --- | --- |
| the plan (`orders` "plan", `anchor`) | +0.036 (+0.016 to +0.056), 8 of 8 | -0.013 (-0.027 to -0.002), 3 of 8 |
| + nearly closed weeks gathered (`close` 0.05) | +0.035 | +0.007 (-0.001 to +0.013), 7 |
| **+ the worth of a whole week in the program (`hull` "round")** | **+0.046 (+0.025 to +0.067), 7** | **+0.021 (+0.007 to +0.038), 6** |
| the plan + the true future in the window | +0.059 (+0.041 to +0.082), 8 | +0.026 (+0.004 to +0.047), 5 |
| the plan + the true future + gathered weeks | +0.058 | +0.041 (+0.027 to +0.056), 8 |
| the full-knowledge plan from the hybrid's trajectory | +0.070 | +0.064 |

Two levers that add up: whole weeks (the plan's structure: +0.015 on Full with or without the true future) and
knowing the window (+0.024 on Small, +0.035 on Full).

**Why the program spreads the fuel, and the fix.** A fab runs only on what its grid delivers above the base load,
and a short week's cell has the fabs off by definition. A unit of the scarce fuel is then worth only the base load
it serves (0.74 bn USD a week at JP), never the lots of a whole week (7.2 bn). Shed load is linear in fuel, so
burning a little every week and burning in whole weeks cost the program the same, and under the gas ration an even
inflow is a fixed point: what arrives is burned. The hybrid's rules build the cycle by hand; the program had no
reason to.

- **"HULL"** (`core.Episode.cell`): a short week as a mix of whole weeks and weeks without the scarce fuel. The
  fabs run for the share `rho <= G_k / cap_k` of a full week's burn that every fuel segment gets: the convex hull
  of the base-first rule along the scarce fuel. The oracle's relaxation lets the fabs run before the base load, the
  plain cell not at all; the hull is between them and can be reached by alternating weeks.
- **Rounding** (`_rounded`): along a grid's weeks the shares are summed; each time they reach the price of one more
  whole week the week is marked. The price is a week's burn, or, for gas under its ration, the stock the ration
  asks for the week before (`psi * ibar`, 1.2 weeks of burn).
- **"SOFT"**: a marked week is the regimes "OFF" and "MID" in one cell, with a price on its shed base load that no
  lot is worth (`SOFT`, 1e8 USD a unit). The fabs then run only when the base load is served in full; the cell
  always has a solution, and the solution is the simulator's (offline: the claimed cost equals the played one).
- A week's step: the cell with the hull in every short week (one solve), the rounding, the exact cell with the
  marked weeks soft (the usual solve).

| the hull's variant | Small | Full |
| --- | --- | --- |
| marked weeks closed hard (`hull` "hard"); the plain cell when that has no solution | +0.047, 7 of 8 | +0.018 (+0.003 to +0.034), 5 |
| **marked weeks soft (`hull` "round")** | **+0.046** | **+0.021** |
| no solve with the hull: weeks marked from the reference's own burn (`hull` "burn") | +0.036 | +0.007 (-0.005 to +0.018) |
| soft, the price for leaving the rules on orders only | +0.042 | +0.018 (+0.005 to +0.031), 7 |
| soft, the rules' orders, no price | +0.040 | +0.020 (+0.004 to +0.036), 7 |

Against the plan without it, paired: Small +0.010 (+0.004 to +0.018), better in 7 of 8; Full +0.034 (+0.020 to
+0.053), better in 8 of 8. In the ledger the sales turn from a loss into a gain on Full (+105, then +14 with
gathered weeks, then -31 bn USD an episode against the hybrid) and grow on Small (-7 to -19). Full, episode 3, JP:
30 whole weeks under the hybrid, 6 under the plan, 33 with the hull; the episode's score 0.559, 0.526, 0.639 (0.618
with the true future in the window). CPU a week outside the container on a busy machine: 2.7 / 3.8 s on Full
(median / 95th percentile), 0.31 / 0.54 s on Small: over Full's budget as it stands.

What the variants say: it is the worth inside the program that pays, not the marking of weeks (no gain without the
solve that has the hull); closing the marked weeks hard loses a week's whole plan when one grid's weeks cannot be
closed (Full, episode 2, week 20: CN's weeks took JP's with them); the three ways to keep the plan near the rules
score the same on Full. Without the price on wafers the plan starts fewer lots that end as thrown-away chips (4.9
against 2.6 M fewer than the hybrid), but a lot cut in a whole week gives its energy to nobody: generation falls by
as much.

**Episodes the hull was not developed on** (it was developed on root 444, episodes 0 to 7; the soft variant, the
same settings). Small 444, episodes 8 to 23: 0.9225 against the hybrid's 0.9035, +0.025 (+0.011 to +0.039), better
in 13 of 16; against the plan without it +0.009 (+0.003 to +0.017), better in 12. Full 444, episodes 8 to 15: 0.9026
against 0.8653, +0.027 (+0.015 to +0.040), better in 8 of 8; the plan without the hull is at -0.025 there (better in
0 of 8), and the hull against it +0.052 (+0.038 to +0.068). The selection set, a separate run by `plan_lab` on a
frozen copy of the folder: Small 111 x 64, 0.8661 against 0.8336, +0.033 (+0.021 to +0.043), better in 57 of 64.

**The ceiling with the hull** (`plan.py starts --hull`, the full-knowledge descent from the hybrid's trajectory,
then rounds of the hull; root 444, episodes 0 to 7): Small 0.920 (0.908 without the rounds), Full 0.909 (0.900). A
lower bound: the descent stays local.

**Not moved, and dropped:** the threshold of `close` (15% for 5%) and gas gathered under its ration: 0.8423, 0.8420,
0.8409 against 0.8421 on Full.

**The end value with a cap** (`Episode._pools`, the agent's `end_weeks`; the cause was found in `plan_lab`). The end
value paid 3.3 M USD for a unit of fuel anywhere past its source, so a plan that orders lifted fuel and parked it at
the straits (Full: 154 thousand units of crude there at the end of an episode against 3 under the hybrid, and CN
went short). Now fuel is worth its value up to a cap. A (grid, fuel) pool is the grid's stock, its terminal's and
the cargo on the way to them, capped at the grid's own stock level `ibar` plus `end_weeks` weeks of burn; whatever
else holds the fuel past its sources is one pool a fuel, capped at `end_weeks` weeks of all the grids' burn. A pool
is one more column of the program; the played cost uses the same formula (claimed and played costs differ by 0.03
bn a window at most).

| variant (all with the hull) | Small 444 x 8 | Full 444 x 8 | Small 111 x 64 | Small 111, episode 63 |
| --- | --- | --- | --- | --- |
| no cap | 0.8844 | 0.8566 | 0.8661 | 0.759 (the hybrid 0.943) |
| 4 weeks | +0.000 (-0.005 to +0.006) | +0.001 (-0.007 to +0.009) | | 0.753 |
| **1 week** | -0.002 (-0.007 to +0.003) | -0.000 (-0.008 to +0.007) | **0.8724, +0.006 (-0.000 to +0.014)** | **0.948** |
| 0.5 week | -0.003 (-0.007 to +0.003) | -0.001 (-0.007 to +0.003) | 0.8736, +0.007 (+0.001 to +0.015) | 0.958 |

Differences are paired with the variant without a cap. At 4 weeks the cap does what it was built for (Full: crude at
the straits 154 to 5 thousand, crude burned 378 to 385 against the hybrid's 388, CN's shed load against the hybrid
+10 to -5 bn USD) and the score does not move: the plan parks gas instead (66 to 85 thousand), which that cap does
not bind. On average the cap is no lever. It is needed for one mechanism: the hull pulls stock into the soft weeks
ahead, and while that stock is paid in full a grid under its ration gets exactly what it burns (Small 111, episode
63: Europe sheds 12% for 8 more weeks; the plan without the hull scores 0.940 there). A tight cap removes it. On the
selection set with 1 week: +0.037 against the hybrid (+0.030 to +0.045), better in 57 of 64; by harm level 0.842 /
0.898 / 0.900 / 0.917 against 0.789 / 0.867 / 0.884 / 0.880; the worst episode against the hybrid -0.025 for
-0.183. One week is the setting; the data do not tell 1 from 0.5.

**The model's three versions, their scores, differences and how they were measured: `versions/README.md`** (in
Ukrainian; the settings are `versions/*.json`, the table is printed by `versions.py`).

**Two models** (built by `build.py`; the first is in `agents/`, the second is built with these `--params`:
`{"share": 0.7, "solve_seconds": 3.0, "orders": "plan", "anchor": {"wafer": 5000, "order": 200000},
"passes": 1, "carry_hints": true, "hull": "round", "end_weeks": 1, "hull_every": 2, "anchor_every": 4,
"hull_only": "tail"}`):

- `anastasiia_plan_hull`: the hull every week (two solves a week), the end value capped at 1 week, no clock. The
  best score; a Full week takes 2.8 s outside the container (median), 4% of the weeks over 4 s.
- `anastasiia_plan_hull_fast`: the same with one solve a week and a clock at 0.7 of the budget. The switches are
  `plan_lab`'s and now live in this lab's source (`hull_every` 2, `hull_only` "tail": in the week of the solve with
  the hull that solution is the week's plan, its first week exact; `anchor_every` 4: the rules alone are rolled out
  every fourth week). With the switches off the source plays as `anastasiia_plan_hull` does, whose folder was built
  just before they came in.

All four agents in one run (`outputs/regime_lab/logs/night_valid.log`; 8 processes side by side, outside the
container), on episodes the hull was not developed on and on the selection sets:

| set | the hybrid | `plan_hull` | one solve a week, no clock | `plan_hull_fast` |
| --- | --- | --- | --- | --- |
| Small 111 x 64 | 0.8336 | 0.8724 | 0.8675 (-0.004 against `plan_hull`, -0.0075 to -0.0017) | 0.8675 |
| Small 444, episodes 8 to 23 | 0.9035 | 0.9225 | 0.9165 (-0.006, -0.012 to -0.002) | 0.9168 |
| Full 444, episodes 8 to 15 | 0.8653 | 0.9049 | 0.9006 (-0.005, -0.011 to -0.0005) | 0.9006 |
| Full 111 x 32 | 0.8707 | 0.9000 | 0.8994 (-0.001, -0.005 to +0.003) | 0.8945 (-0.006, -0.015 to +0.002) |

One solve a week costs up to 0.006 on both networks. The clock changes nothing on Small; on Full 111 it took another
0.005 in this run, a number that depends on the machine's load, since the clock cuts a solve that is late.

**Open after the hull:**

- Which knowledge of the window pays on Full (gathered weeks + the truth about one group, paired with gathered
  weeks): edge capacities +0.016 (+0.007 to +0.025), straits +0.013 (+0.005 to +0.022), plants and fabs 0.000,
  everything +0.035. Supply and demand were not measured apart. Edges and straits are what to forecast.
- The second solve a week on Full: the hull from last week's solution moved on by a week, or every other week.

## Not done

- A regime chosen away from a border (a grid's week switched on or off with the fuel to pay for it): the four
  episodes with 27 to 95 bn left to the bound.
- The straits' default release when the throughput binds is written as the share each lane passed in the replay,
  and a plant's split between two products is linearised: exact in the replayed point only.
- A forecast of edge capacities; scenarios.
- Full inside 4 s: a warm start between passes (only bounds change), a shorter window with an end value.
- The scoring container's clock, the import check (`sbf check`), a folder without the lab's paths.
