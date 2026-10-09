# Decisions under uncertainty: what comes after certainty-equivalent MPC

Brainstorm notes, lens "decisions under uncertainty", 10 October. Read-only pass: nothing in the repository was
changed, nothing was played. Base model throughout: `agents/anastasiia_plan_hazard` (0.8989 Small / 0.9017 Full on
root 222). "Measured today" means I computed it from kept files with one short process; "not checked" means I did
not. Conversion used everywhere: 0.001 of score is 1.03 bn USD an episode on Small and 3.4 bn on Full (mean room
1 032 and 3 350..3 860 bn in `hub/eval/records/anastasiia_plan_hazard.json`).

## 0. Bottom line

1. **The score is a population mean, so the right attitude to risk is none.** Formula (56) of the package
   (`shockbench_flow/scoring/rss.py`): RSS = sum_s p_s mean(saved)_s / sum_s p_s mean(room)_s, with p_s the generator's
   own harm quantiles (0.50 / 0.30 / 0.15 / 0.05). That is a post-stratified estimate of
   E[naive - agent] / E[naive - clairvoyant] under the generator. Every dollar of every population episode counts the
   same. A hedge is worth P(event) x payoff - premium in USD, nothing else. CVaR (the lambda term of P3 in
   `paradigm_lab/SPEC.md`) is not supported by the score.
2. **On Small the information pools left for this lens are small and measured:** ends of running events +0.009, of
   which +0.0059 is taken (`w50as`); onsets of new capacity cuts +0.0006 (edges) and -0.0001 (straits) on the old base
   (`plan_lab/README.md`, table of the window's ceiling). Nothing below about onsets can be worth more than about
   +0.001 on Small.
3. **On Full the pools are larger and not measured from the current base:** onsets +0.011 (+0.004..+0.020, 8 episodes,
   base `plan_hull`), ends +0.013 on that base with about +0.006 taken since. Measured today: new events carry 25 % of
   an episode's fuel cut edge-weeks on Full against 14 % on Small. Every idea below is a Full idea first.
4. **The rate of new events is hardly predictable from the week-1 state; only the two weeks after an onset are.**
   Measured today on root 333 (section 1.3). So "state-dependent safety stock by regime" has little to modulate, and a
   cluster alarm comes later than the 2 to 5 weeks a fuel lane needs: it can redirect cargo, not build stock.
5. **Next solution type, as I see it: not a scenario tree** (the pool does not pay for the columns on Full) **but a
   single LP with first-order option prices taken from the known generator** (a reserve counted in whole-week kits, a
   priced single-contingency reserve, a front-load price on announced cuts), **plus one-sided actions the simulator
   bounds**, **plus a measuring tool that removes events from a scenario** so that a hedge's premium and payoff are
   measured apart. The package ships that tool (`omega/injected.py`, `build_omega`); nothing in `lab/` or `hub/` uses it.

## 1. What I measured today

Scripts lived in a scratch directory outside the repository. Inputs: `hub/eval/records/anastasiia_plan_hazard.json`,
`outputs/hazard_lab/anatomy/anatomy_20261009_195236.pkl` (root 333, Small 300 episodes, Full 100),
`outputs/hazard_lab/play/w50as_{s,f}_*_444.pkl`.

### 1.1 Levels and the score (root 222 records, reference costs and the kept model's costs; nothing fitted)

| set | level | episodes | mean room, bn | mean gap to clairvoyant, bn | RSS of the level | weight x (1 - RSS) |
| --- | --- | --- | --- | --- | --- | --- |
| Small 222 | 1 | 126 | 1 032 | 106.6 | 0.897 | 0.0516 |
| | 2 | 80 | 1 038 | 106.5 | 0.897 | 0.0308 |
| | 3 | 37 | 928 | 91.7 | 0.901 | 0.0148 |
| | 4 | 13 | 882 | 63.8 | 0.928 | 0.0036 |
| Full 222 | 1 | 69 | 3 350 | 311 | 0.907 | 0.0465 |
| | 2 | 37 | 3 421 | 323 | 0.905 | 0.0284 |
| | 3 | 19 | 3 863 | 413 | 0.893 | 0.0160 |
| | 4 | 3 | 3 316 | 480 | 0.855 | 0.0072 |

- A local set of the first n episodes of a root has the levels in the generator's proportion (126 / 80 / 37 / 13 of
  256), which is the proportion of the weights: per population episode a dollar is worth the same in every level. On
  the board the levels have equal counts, so one board episode of level 4 counts a tenth of one of level 1, because it
  is drawn five times too often and level 1 half as often as in the population.
- Consequence for hedging: nothing changes with the level. Half of the weight on the calm half is the population, not
  an emphasis. An agent should not carry a different posture by level, and a parameter should be chosen on sets in
  the generator's proportion, as the team already does.
- On Full the stormy levels are the weak ones (0.893, 0.855; only 19 and 3 episodes), on Small they are the strong
  ones. That fits item 3 of the bottom line, and is too few episodes to be more than a pointer.

### 1.2 What of an episode's fuel cut is new (root 333; edge-weeks of fuel edges with a cut capacity, plus fuel edges at a shut strait)

| | Small | Full |
| --- | --- | --- |
| carried in, edge-weeks an episode | 859 | 2 337 |
| new (onset inside the episode) | 142 | 797 |
| new share | **0.14** | **0.25** |
| new, by type: sanction / militarised closure / conflict | 43 / 42 / 32 | 281 / 220 / 192 |
| new, by type: slowdown / weather / stoppage / derived closure | 8 / 10 / 4 / 2 | 39 / 37 / 17 / 11 |
| short Poisson events in the new part | 16 % | 12 % |

- New events an episode that cut a fuel edge or shut a fuel strait: sanctions 1.36 Small and 4.47 Full, militarised
  closures 0.47 and 1.10, conflicts 0.25 and 0.61, weather closures 1.17 and 3.18, stoppages 0.51 and 1.59, slowdowns
  0.27 and 0.79; energy shocks 1.47 and 4.70.
- **84 to 88 % of the new fuel cut is long events** (sanction side cuts, militarised closures, conflicts). A buffer of
  one or two weeks does not outlast them; what foresight can save is the transition: the weeks until the supply of
  another lane arrives, and cargo already sent towards the cut.
- Announced a week or more ahead (Small, `hub/findings/data/event_stats_small.md`): 47 % of sanctions, 65 % of
  militarised closures. By edge-weeks that is about 34 % of the new fuel cut on both networks, if Full announces at
  the same rates (not checked).

### 1.3 How predictable the onsets are (root 333)

By episode, quartiles of the carried fuel cut (what week 1 shows):

| quartile of the carried cut | Small: new capacity events | Small: new fuel edge-weeks | Full: new capacity events | Full: new fuel edge-weeks |
| --- | --- | --- | --- | --- |
| 1 (calmest) | 5.0 | 143 | 13.8 | 661 |
| 2 | 4.4 | 104 | 13.5 | 625 |
| 3 | 5.1 | 140 | 15.4 | 955 |
| 4 | 5.8 | 181 | 15.0 | 946 |

Correlation of the carried and the new fuel cut 0.11 (Small) and 0.22 (Full); of carried and new events of the
conflict block 0.07 and 0.14. The calmest half of the episodes has 43 % and 40 % of the new fuel cut. **The week-1
state does not say how many new events an episode will have.**

By region (what the region shows in week 1, new events of the conflict block in it over the episode):

| region in week 1 | Small: region-episodes | new events | closures and conflicts | Full: region-episodes | new events | closures and conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| nothing of the block running | 3 538 | 0.19 | 0.05 | 1 254 | 0.47 | 0.09 |
| an event of the block running | 187 | 0.28 | 0.11 | 81 | 0.86 | 0.40 |
| a conflict running (war) | 175 | 0.32 | 0.25 | 65 | 0.80 | 0.57 |

A region at war has five to six times the rate of new closures and conflicts, but such regions are 4.5 % of
region-episodes and hold 18 to 20 % of those onsets; the weekly chance stays near 0.5 %. For the policy block the
same holds: a region with 3 or more carried events has 0.22 new sanctions with a capacity cut on Small against 0.06
with none, 0.62 against 0.17 on Full.

The cluster, the only strong signal:

| after a new onset is seen in a region | Small | Full |
| --- | --- | --- |
| conflict block: another new onset of the block there within 1 / 2 / 4 weeks | 0.19 / 0.28 / 0.32 | 0.18 / 0.25 / 0.28 |
| the same by the base rate | 0.004 / 0.008 / 0.015 | 0.005 / 0.010 / 0.019 |
| policy block: another within 1 / 2 / 4 weeks (with a capacity cut) | 0.16 / 0.22 / 0.26 (0.05 / 0.08 / 0.11) | 0.15 / 0.20 / 0.25 (0.07 / 0.10 / 0.13) |
| share of new militarised closures that follow such an onset within 1 / 2 / 4 weeks | 0.22 / 0.31 / 0.35 | 0.15 / 0.20 / 0.25 |
| share of the new fuel cut (conflict and policy blocks) that follows one | 0.19 / 0.24 / 0.27 | 0.14 / 0.17 / 0.20 |
| region-weeks under such an alarm an episode, conflict block (2 weeks) | 3.9 of 676 | 11.0 of 1 456 |

The lift is 35 to 50 times at the level of a region. At the level of one strait it is about 0.19 x 0.375 (the share
of closures) over the region's adjacent straits: 2 to 7 % a week. That pays only for hedges that cost next to
nothing, and the alarm's lead (one or two weeks) is shorter than a fuel lane (source to strait 1 to 2 weeks, strait
to terminal 1 to 3).

### 1.4 What an onset costs the played agent (event study on kept plays; noisy, one run, overlapping events inflate it)

Weekly shed load of `w50as` around the week an event starts, minus the mean of the four weeks before, minus the same
statistic at weeks with no new capacity event near; bn USD an event, plus or minus one standard error.

| type | Small 444 x64: first 4 weeks | next 8 weeks | Full 444 x16: first 4 weeks | next 8 weeks |
| --- | --- | --- | --- | --- |
| port stoppage (42 and 23 events) | +14.5 +- 10.8 | +38.9 +- 17.4 | +20.1 +- 15.4 | +57.1 +- 29.8 |
| weather closure (57 and 40) | +1.4 +- 8.7 | +16.8 +- 22.4 | +7.8 +- 20.2 | -32 +- 46 |
| energy shock (72 and 69) | +71 +- 12 | -6 +- 25 | +209 +- 30 | +160 +- 45 |

- A stoppage of 1.6 weeks costs the agent some tens of bn, most of it five to twelve weeks later, when the hole in the
  pipeline reaches the grids. The whole onset pool for edges on Small is +0.0006 (0.6 bn an episode), so nearly all
  of that damage is throughput that foresight cannot buy back either. The same strikes are in both columns (Small and
  Full of one root share them).
- Energy shocks are the largest single item of shed load (about 117 bn an episode on Small in the first four weeks),
  and the capacity loss itself is common to the agent, the naive rule and the clairvoyant. On Full the eight weeks
  after carry +160 +- 45 where the law's own tail of long shocks explains about +75; the excess is 1.8 standard
  errors and Small shows none. `grids: "step"` on the same 16 episodes was +0.0004 (-0.0021..+0.0027), which caps
  what a forecast of the shock's end can take at about +0.003. Left open; the tool of idea 0 settles it exactly.
- Sanctions, closures and conflicts are not in the table: their windows overlap each other, and the sums mean nothing.

### 1.5 Straits as hubs (instance, both networks)

Queued tanker cargo can be released onto any out edge of its strait that carries the fuel. Gas at Hormuz has 5 heads
on Small and 6 on Full (Taiwan terminal, India terminal on Full, and Malacca, Taiwan, Suez, Cape onward); gas at
Malacca 3 (Korea, Taiwan terminals, Taiwan strait); crude at Malacca 3 and 4 (Japan, Korea, China terminals, Taiwan
strait); crude at the Taiwan strait 2 and 3. Travel to a strait 1 to 2 weeks, after it 1 to 3. So a queue at Hormuz
or Malacca is a reserve that two to four grids share, one to two weeks closer than the source. Nuclear fuel has no
strait and no hub.

### 1.6 The package can play an edited scenario

`shockbench_flow/omega/injected.py`: `build_omega(inst, events, entropy=, episode=, split=, params=)` returns a
complete scenario for an explicit list of events (`InjectedEvent`: type, target, onset, duration, severity, region,
counterpart, commodity, rate, restoration); `event_free` gives the scenario with none. Carried events are allowed
(negative onset). The built scenario has the same demand stream and no messages or warnings; in the model's folder
only `lp_part.py` reads announcements (`pending_prohibitions`), nothing reads `warning` or `messages`. Not checked:
that the events of a generated scenario survive the round trip to the cent, and how the environment is reset from a
given scenario (`dynamics/env.py` accepts one).

## 2. The pools this lens draws on

| pool | Small | Full | source |
| --- | --- | --- | --- |
| ends of running events, capacities and straits | +0.0090 (+0.0068..+0.0115) from `hull3`; taken +0.0059 | +0.0128 from `plan_hull`; taken about +0.006; not measured from `hull3` | `hazard_lab/README.md`, `plan_lab/README.md` |
| of that, what is left by type | sanctions +0.0014, militarised closures about +0.001, conflicts +0.0002 | not measured; by edge-weeks with an end inside the episode Full has 1.7 times Small's exposure per unit of room | same; `GENERATOR.md`, table e |
| onsets, all fields | +0.0030 (+0.001..+0.006), 16 episodes | +0.0112 (+0.004..+0.020), 8 episodes | `plan_lab/README.md`, base `plan_hull` |
| onsets, edges only; straits only | +0.0006; -0.0001 | not measured | same |
| everything in the window | about +0.017 over `hull3`'s score (the old oracle's level, 0.9224, on the same 16 episodes; not played from `hull3`) | +0.026 from `plan_hull` | `hazard_lab/REVIEW.md`, section E |
| the whole future, best executable plan | +0.032..+0.035 | about the same | `hub/FINDINGS.md` |
| fuel mix, not information | shortage +16.7 and shed +11.9 bn an episode on 4 episodes | not measured | `cycle_lab/README.md` |

The difference between a planner with information and one without is the sum of what perfect information is worth
and what a hedged plan is worth. A hedge can take at most the pool, and takes much of it only where carrying the
hedge is nearly free.

## 3. The six questions

**State-dependent safety stock driven by the hazard.** The rate is not predictable at the horizon stock needs
(section 1.3): by episode not at all, by region a factor of five on 4.5 % of region-episodes, by cluster a factor of
40 for two weeks. Stock takes 2 to 5 weeks to build. So the hazard cannot time a build-up; it can set a constant
small reserve where the reserve is free (idea 1) and steer fast controls, the release at a hub and the valve, in the
two weeks of an alarm (ideas 2 and 6). Expectation for the modulation alone: under +0.001 on Full, nothing on Small.

**Risk pooling at straits with re-routing at release.** The physics is there (section 1.5), and the plan already
parks selectively; the rules' buffer of 2 to 3 weeks at a strait is inherited through the anchor on orders
(`regime.json`: 200 000 USD a unit of deviation from the rules' orders). What a point forecast cannot see is that one
queue covers several grids' single failures. That is a credit in the reserve rows of idea 2, capped at 2 to 3 weeks
because parking of 4 weeks and more was measured to block the oil behind the gas. It draws on the onset pool: about
nothing on Small, part of +0.011 on Full.

**A decision rule (affine or piecewise recourse) inside one LP.** With jumps as the uncertainty an affine rule is a
recourse block for each contingency. On Full the fuel transport columns of 8 weeks are roughly 6 to 8 % of the cell
(my estimate from 50 101 columns, not measured), so each contingency costs 0.1 to 0.3 s where the 95th percentile of
the week is 3.14 s of 4. It would buy, beyond a priced reserve, the exact worth of re-routing and of cargo placed in
front of a cut that may end. Its pool is the same as idea 2's. Idea 5 gives the form; I would not build it before
idea 2, its price form at 1 % more rows, shows a gain on Full.

**Onsets that are announced.** By events an episode on Small: 0.73 pending prohibitions (0.35 real, lead 5 weeks),
0.61 military threats (0.46 real, lead 1 week, 3 at the third quartile), 3.71 sanction threats (half real, lead 0 for
a quarter and 13 at the median). About a third of the new fuel cut is announced. `pending: true` wrote the
prohibition into the forecast as certain and measured -0.0005; it never used the capacity cut on the dyad's other
edges, which is the part that moves fuel. Idea 3 uses the announcement as a price on the weeks before the date and
leaves the forecast alone.

**Whether the level weights change what should be hedged.** No (section 1.1).

**Robust for free, as over-asking was.** I looked for other places where the simulator bounds the downside. The clips
are pro rata at the edge, the stock, the strait's throughput and the fleet pool; of these only the strait's
throughput is still unused (`ask_scale_out` in the working tree covers releases onto a cut edge; a release limited by
a militarised closure's throughput is the same clip). Stock and fleet are known exactly at the instant of the
action, so there is nothing to over-ask against. Two requests on two lanes from one stock are shared pro rata, so
they cannot be made conditional. I found no new free move of that kind. The nearest things are cheap, not free: the
kit reserve of idea 1 (small volumes of the small gate fuel) and the release at a hub in a week of alarm.

## 4. Ideas, ranked by expected value

### 0. Take events out of the scenario: premium and payoff measured apart (the gate for everything below)

1. **Formulation.** A lab harness over `omega/injected.py`: read a generated scenario's events
   (`marks.read_events`), edit the list, `build_omega`, play. Three uses. (a) The frozen world: drop every event with
   an onset inside the episode and stretch every carried event past the last week; the forecast "as observed" is then
   exact. (b) One kind back in: frozen plus the ends of carried events, plus strikes and weather, plus energy shocks,
   plus new sanctions, plus new closures and conflicts. (c) For a hedge: its premium is cost(hedge) - cost(base) in
   the frozen world, its payoff is the same difference in the world with the events.
2. **Mechanism.** Not a lever. The oracle measurements give the planner information and keep the world; this removes
   the uncertainty from the world and keeps the planner. RSS(frozen) - RSS(real) is everything uncertainty costs this
   planner, by kind, on the same episodes; 1 - RSS(frozen) is plan quality and the clairvoyant's relaxation.
3. **Not dead:** nothing like it in `hub/tried/` or `paradigm_lab/`; no file in `lab/`, `hub/` or `src/` calls
   `build_omega`.
4. **Size.** No score of its own. It replaces paired runs, where one episode moves by about 0.009 of score for no
   reason (the placebo terms of `evolve_lab`), with pairs that differ by one event.
5. **Falsifier of the lens.** If RSS(frozen) - RSS(real) is under +0.005 on Full, close ideas 2 to 6. The version with
   the existing harness, a few lines in `_events_told`: tell the events that have not started and start within `look`
   weeks (1, 2, 4, 26), by type, from `plan_hazard`, Full 444 x16 and Small 444 x64; and play the built folders
   `ends_all_f`, `ends_short_f`, `ends_energy_f`.
6. **CPU and risk.** Offline. References of an edited scenario cost 5 s (Small) and 68 s (Full) an episode. Risk: the
   round trip of events is not checked; a built scenario has no announcements, so idea 3 needs the generated one.

### 1. The reserve counted in kits: a whole week needs every gate fuel, so stock the scarcest one

1. **Formulation.** For a grid g with fabs and its gate fuels K_g (weekly burn above the top of the output, the test
   `cell(..., gate=True)` already has), let S[g,k,t] be the fuel at the grid and its own terminal at the end of week t
   (the pools of `plan_core._pools` without cargo in transit) and cap_k the weekly burn. One new column kit[g,t] in
   [0, B], rows kit[g,t] <= S[g,k,t] / cap_k for each k in K_g, cost -rho x V_g x kit[g,t]. V_g is what a whole week
   of the grid's fabs sells for (the hull has it), B is 2, rho is the weekly chance that one of the grid's gate lanes
   is interrupted: about 0.01 to 0.02 from the generator's rates (strike 0.1 a region a year, weather 0.2 a strait a
   year, the two blocks' base rates), more under an alarm or a threat. The marginal worth of the scarce fuel is then
   rho x V_g / cap_k, to be kept under about 0.5 mn USD a GWh, an eighth of the price of shed load.
2. **Mechanism.** Stock of the smallest gate fuel in weeks of burn. A fab runs only when every gate fuel is there, so
   the week is a series system and its weakest link is cheap to stock (oil in Japan is 900 a week against 5 400 of
   gas). It pays when a lane of the small fuel blinks, and it is the same quantity `cycle_lab` found missing with no
   event at all: the agent's small segment at 0.06 of its cap against the teacher's 0.87, 49 bn in one episode, with
   free routes all episode.
3. **Not dead.** P5 made the gate fuels equal and the cell infeasible; a free column cannot. `evolve_lab` a1 and a3
   were prices on one fuel's stock (a3 only before a marked week, half of its code inactive, -0.0003 on 128
   episodes); the hook there takes a price vector only and cannot write a minimum over fuels. `stock_floor` (-0.002)
   was a floor on every fuel at the rules' level. The margin on capacities removes throughput; this removes none.
4. **Size.** Against onsets alone: under +0.001 Small, under +0.002 Full. The larger part is the fuel-mix gap, which
   is the objective's, not information: 28.6 bn an episode on 4 Small episodes, one episode holding 49 of the 114.
   If a tenth to a fifth of it generalises: +0.003 to +0.005 on Small; on Full five grids have such a fuel and whole
   weeks are worth +0.034 against +0.010, so the same order (not checked). Wide: the nearest relative measured nil.
5. **Falsifier.** No new play: on the 4 episodes of `cycle_lab` compute min over gate fuels of S / cap for the agent
   and the teacher. Prediction: the teacher's is higher in the four weeks before each week the agent sheds and the
   teacher does not. If not, drop it. Then rho 0 (to the cent with the base), 0.01, 0.05 on Small 444 episodes 0 to 3,
   where episode 3 must give back a visible part of 49 bn; then 128 fresh episodes on each network.
6. **CPU and risk.** Full: 182 columns and about 550 rows on 50 101 columns, under 0.03 s. Risk: any new price moves
   episodes by 0.009 each way, so nothing under 128 fresh episodes is a reading; overlap with the owner of the open
   form of P5.

### 2. A priced reserve against each single failure, with the hub's queue shared between grids

1. **Formulation.** For each grid, gate fuel and contingency c of a short list (each strait on the fuel's lanes in
   use, the port region of the terminal and of each source, a dyad with a pending prohibition, a region under a
   cluster alarm), for weeks t = 1..8: a column short[g,k,c,t] >= 0 with the row
   short >= (arrivals of k at g through c in the d_c weeks from t) - (S[g,k,t-1] above the ration threshold) -
   (queue of k at a strait from which g is within d_c weeks and which c does not touch), cost
   lambda_c(t) x v x short. d_c: 1 week for weather, 2 for a stoppage, for a long cut the travel time of the next
   lane (2 to 5 weeks). v: 4.1 mn USD a GWh. lambda_c: the generator's table (`task_generator`, written to a json
   beside the agent), times the alarm of section 1.3 (0.19 next week for the region, about 0.02 to 0.07 for one
   strait), 0.72 for a strait under a military threat, 0.48 x P(side cut) for a pending prohibition. The same queue
   column stands in several grids' rows: one failure at a time.
2. **Mechanism.** Where fuel waits, and how much a grid depends on one element. In calm weeks the price is 0.2 to 1 %
   of the price of shed load, so it moves stock only where lanes and sources have slack. It pays in the transition
   after a new cut and in weeks of alarm.
3. **Not dead.** The margin u(1 - k sigma sqrt t) takes capacity from every edge every week against a jump that
   happens in 1 % of edge-weeks; here no capacity is removed and the price is the jump's own hazard. `evolve_lab` d1
   and d2 priced flow on elements the forecast itself has falling later, which "as observed" never has (d1 was to the
   cent with the base on Small); a2 and y4 priced against queues. The uniform mix over routes (-283 bn an episode)
   forced diversification; here the program may hold stock instead.
4. **Size.** The onset pool: Small about nothing (+0.0006 and -0.0001 measured for edges and straits), Full a fifth
   to a third of +0.011, +0.002 to +0.004, if that pool survives the re-measurement of idea 0.
5. **Falsifier.** The onset oracle with `look` 2 and 4 from `plan_hazard` on Full 444 x16. Under +0.003 at `look` 4:
   drop this and ideas 3, 5, 6.
6. **CPU and risk.** Full: about 640 rows and columns (1.3 %), under 0.05 s. Risk: the queue credit invites parking;
   cap it at 2 to 3 weeks of the grids' burn.

### 3. An announcement as an option: a price on the weeks before the date, not a forecast

1. **Formulation.** (a) A military threat on an open strait: this week's tanker releases there are asked up to the
   queue (free where the throughput binds), and the cell gets -p x v on fuel passing the strait in weeks 1..3, p =
   0.72 times the chance the effect falls in those weeks (lead 0 / 1 / 3 / 9 weeks at the quartiles and the ninth
   decile). (b) A pending prohibition on a dyad at week w: -p x pi_e on fuel flow over the dyad's other edges in the
   weeks before w, where pi_e is the last solve's dual of that edge's capacity (zero on a slack edge, so the price
   limits itself) and p = 0.48 x P(the dyad's fuel edges are cut at w, given an entry). The forecast is untouched;
   lots and wafers see nothing.
2. **Mechanism.** Fuel through an element in the weeks before it is cut for the rest of the episode; cargo out of a
   strait before it closes for a median of 31 weeks.
3. **Not dead.** `pending: true` (-0.0005) changed feasibility for every block at 52 % decoys, and held only the
   prohibition, not the capacity cut. A price on fuel timing cannot add lots, which is how the joint program lost on
   Full.
4. **Size.** About 34 % of the new fuel cut is announced. Small: under +0.0005. Full: +0.0005 to +0.0015.
5. **Falsifier.** No play: from `anatomy.py` and the message tables, on Full root 333, the fuel edge-weeks of real
   announced events and P(side cut given an entry); under 5 % of the new fuel cut closes it. Then the oracle that
   tells only events with a live thread.
6. **CPU and risk.** None; the chance of a military threat falls with its age (0.72 to 0.62 over 8 weeks).

### 4. Read the score as a mean: mean over scenarios, weights from the generator, no tail term

No lever of its own. If P3's joint program is built (idea 5), its objective is the probability-weighted mean and the
weights are the generator's chances, not a tuned lambda. The evidence that "everything that worked reduced a tail" is
about the model's error, not the score: the cure for that is the replay as the judge, which the model has. A
parameter that helps stormy episodes at a cost to calm ones is to be judged on the plain paired mean in USD.

### 5. Recourse blocks for the one or two contingencies that matter this week

1. **Formulation.** After the exact solve, rank contingencies by lambda_c x sum over (e, t) of pi[e,t] x (capacity c
   would remove): the duals come with the solve. For the top one or two, a second block of the fuel transport columns
   only (orders, releases, valves, stocks, queues) of the grids touched, weeks 2..9, on c's network; week 1 and all
   lots and wafers are shared; objective (1 - sum p) J_0 + sum p_c J_c. An end of a running cut is a contingency like
   any other ("this cut is over from week tau"), with p from the law of its kind and age.
2. **Mechanism.** The same as idea 2, plus what a priced reserve cannot value: where to re-route, and cargo placed in
   front of a cut that may end (new militarised closures end within 26 weeks with chance 0.47 at age 1; 1.1 of them an
   episode on Full).
3. **Not dead.** The joint program with two forecasts let the second block start lots (+0.49 mn lots, -0.17 mn
   leading-edge chips sold on Full); here the second block has no lots. Its second forecast was one optimistic date for
   every cut; here a block is one event with its own chance.
4. **Size.** Small: nothing beyond the hull was measured for the joint form, and the onset pool is about zero: 0 to
   +0.001. Full: +0.001 to +0.003.
5. **Falsifier.** Idea 2 on Full first. Then the same program with the scenario's true chances as weights; if that is
   nil, no estimate of the hazard will do better.
6. **CPU and risk.** About 6 to 8 % more columns a block on Full (not measured): one block fits, two do not, unless
   the clock shortens the window.

### 6. The route price that is already confirmed, with the hazard of the week in place of a constant

`evolve_lab` confirmed two terms at +0.002 on both networks; one is "of parallel routes the longer costs 1 % of the
cargo's worth a week". FINDINGS has capacity changing in about 1 % of edge-weeks, so the constant found by search is
of the size of the jump hazard, and its other half is a price on burning fuel that could wait for a whole week: both
are option prices. The step that follows: the price of a week of commitment on a route is the sum of the hazards of
its elements that week (calm rates from the generator; 2 to 7 % for a strait of a region under alarm; 0.72 under a
threat). Objective only, so it fits that lab's hook and protocol. Size: the alarm covers 4 to 11 region-weeks an
episode and 14 to 27 % of the new fuel cut follows one; under +0.001 on Full, nothing on Small. Falsifier: 512 fresh
episodes there; under +0.0005 closes it.

### 7. What is left of the one-sided moves

The release limited by a militarised closure's throughput (same clip as the edge; `ask_scale_out` covers releases
onto a cut edge), and default release in place of a zero override at a shut strait. `REVIEW.md` D1 sized the whole
family at +0.001 to +0.003; `ask_scale` took +0.0009 and +0.0019. Ceiling: the `look: 1` oracle. Expected remainder:
+0.0002 to +0.0005 Small, +0.0005 to +0.001 Full, at no CPU.

## 5. Order of runs, after the formal evaluation is over

1. No play: kit counts of agent and teacher on the 4 `cycle_lab` episodes (idea 1); the table of announced fuel
   edge-weeks on Full (idea 3).
2. `ends_all_f`, `ends_short_f`, `ends_energy_f` (built, not played) and the onset oracle with `look` 1, 2, 4, 26 by
   type on Full 444 x16 and Small 444 x64: the pools of this lens from the current base.
3. The harness of idea 0: round trip of one generated scenario to the cent, then the frozen world on Small 444 x32
   and Full 444 x8.
4. Ideas 1 and 7 (no dependence on the pools), then 2 and 3 if step 2 leaves at least +0.003 at `look` 4 on Full.

## 6. Not checked

- Every size of section 4 is an estimate; none was played.
- Announcement rates on Full; P(side cut on fuel edges given a pending entry); the share of fuel transport columns in
  the Full cell.
- The onset pools are from `plan_hull` on 16 and 8 episodes; the model has changed twice since.
- The event study is one run of one model; the sanction, closure and conflict rows were dropped as unreadable.
- Region of an event in the anatomy records is the event's own region; children in other regions are not counted, so
  the cluster numbers are slightly low.
- Whether the plan on Full ships nuclear fuel as early as the rules do (the rules' early shipping was worth +0.013 on
  96 episodes against routes that close for good); the end credit of the window should keep it, I did not look.
