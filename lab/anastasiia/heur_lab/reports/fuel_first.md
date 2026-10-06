# fuel_first: fuel ordered by need, within what the lanes carry, and concentrated in time

Written by the lead from the subagent's hand-back (the subagent could not write files in `MAIN`); its raw records are
in `fuel_first_lab_out/records/` (the table of all 130 variants against `pull`, per-episode costs, `diag.py` output
before and after, the `sbf check` logs) and its scripts in `fuel_first_lab_out/tools/`. Agent:
`outputs/heur_lab/agents/fuel_first/agent.py`. The two scores in the first table were reproduced by the lead's own runs (0.6998 on Small, 0.7078 on Full); the rest
are the subagent's. `records/table_variants_by_stage.txt` is the readable table of every variant.

## Result

| set | pull | fuel_first | difference | 90% interval |
| --- | --- | --- | --- | --- |
| Small, root 111, 64 episodes | 0.4968 | 0.6998 | +0.2030 | +0.1746 to +0.2299 |
| Full, root 111, 32 episodes | 0.3981 | 0.7078 | +0.3096 | +0.2685 to +0.3576 |

By harm level on Small: 0.648, 0.744, 0.753, 0.744. The 64 episodes of Small were also the tuning set. The structure
was decided on episodes 0 to 31 and scored on 32 to 63, where every step gained (pull 0.518, first version 0.586,
with time concentration 0.641, two order passes 0.655, final structure 0.682); the last parameter refinements saw
both halves, so the 64-episode figure is slightly optimistic. `sbf check` passes on Small (worst week 11 ms) and on
Full (33 ms). Non-fuel slots are `pull`'s, entry for entry.

## What it does

`FuelRules(config, FUEL).fill(observation, flows)` overwrites the fuel entries of `flows` and nothing else.

1. **Terminal-to-grid valve.** The grid closes every week at the lng rationing threshold plus 0.3 weeks of burn; the
   valve moves what that takes. Crude, which starts at the terminals, reaches the grids this way.
2. **Time concentration.** A grid with fabs whose lng inflow is below 0.9 of its burn does not burn a steady partial
   amount: it holds lng at the terminal, primes the grid with the threshold when terminal plus grid stock cover the
   threshold and two weeks of burn, then runs complete until the terminal is dry. Shed is linear in the fuel burned,
   so the off weeks cost what the steady shortfall would, and the on weeks power the fabs. Not in the last 16 weeks.
   Only grids where the sliver is worth it pulse (TW, KR, JP on Small; EU never).
3. **Orders from the sources.** Order-up-to on the inventory position: stock at the grid and its terminals, cargo in
   transit (from `pipeline.*`), cargo waiting at straits as far as the strait can pass it in 4 weeks. A request is what
   the lane can still carry this week (tightest edge, strait throughput, source stock, the fleet slack of duplicate
   lanes), so requests are executed as asked. Two passes: every grid's need to burn over the lead time first (crude
   before lng), then a top-up of 4 weeks from what is left, so one grid's buffer never takes fuel another must burn.
4. **Nuclear fuel** only when stock plus cargo in transit will not last the weeks that remain (never on Small).

## What mattered (each removed from the version one step before the final, 0.6963, 64 episodes)

| removed | score | change |
| --- | --- | --- |
| requests ignore what lanes carry | -0.061 | -0.757 |
| position ignores cargo in transit and at straits | 0.590 | -0.106 |
| no top-up pass | 0.604 | -0.092 |
| no time concentration | 0.617 | -0.080 |
| lng before crude | 0.679 | -0.017 |
| crude not held while the grid is off | 0.693 | -0.004 |
| nuclear fuel at the lane maximum | 0.692 | -0.004 |

On Full the time concentration is worth 0.708 against 0.600.

## Did not matter or hurt

The order of the grids (reversed: -0.005); the size of the top-up (2 to 6 weeks) and of the safety stock (0.2 to 0.4
weeks; 0 costs -0.037); lane order by cost (-0.006); pulsing at EU; feeding a grid only when its fabs hold wafers
(-0.014 to -0.02); yielding to grids that can be completed (-0.002); a dynamic end-of-horizon rule (-0.005); counting
all queued cargo against a strait's throughput (-0.055).

## Physical effect (12 episodes of root 111: pull, fuel_first, clairvoyant)

- Shed, GWh per week: 10,209, 9,809, 9,564. TW sheds in 79%, 32%, 43% of weeks.
- Lots per week, all fabs: 116,849, 205,851, 248,670. TW leading fab: 13%, 36%, 32% of capacity; KR: 13%, 25%, 30%.
- Cost per episode, USD bn: shortage 1,124, 995, 747; shed 2,190, 2,104, 2,052.

## For the chip rules

- In weeks a grid runs without shed its fabs start only 0.5 to 0.8 of capacity, with wafer stock below capacity in 52
  to 94% of those weeks, while wafers pile up in the grid's off weeks: stock wafers before a run.
- 18% of TW's fab energy goes to the mature fab (11 M USD per GWh against 25 M at the leading fab); the split follows
  what each fab asks for, min(capacity, wafers on hand).

## Corrections to the lab README

`graph_now.kappa.*` already includes `open` (using both counts a closure twice); the tanker fleet slack is binding for
`pull` (cap 4,132 a week on Small) and planning at exactly 100% of it is best; "6 to 10 times VOLL" holds for chip_le
fabs only (2.8 times at the mature fabs); the same convexity holds in time, which is the largest rule after
capacity-aware lanes.
