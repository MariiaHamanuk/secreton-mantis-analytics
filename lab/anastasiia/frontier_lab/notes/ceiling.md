# Brainstorm notes: devil's advocate on the ceiling and on where the money is

10 October. Read-only analysis of files already on disk; nothing was played, no LP was solved. Every number below is
marked **measured** (computed here from the named file, or quoted from a repo document) or **inference**.

Sources used
- `hub/eval/records/*.json` (Small 222 x256, Full 222 x128; per-episode costs of the model, naive, clairvoyant, level).
- `outputs/plan_stats/20261006_040041/episodes.npz` (Small 444 x40: clairvoyant LP and base-first MILP, weekly lots,
  fab energy, shed, served, stock, sent, capacity, bans, `mip_gap`).
- `outputs/heur4/lead/plan_full_444_x20/episodes.npz` (Full 444, 19 episodes = 0-19 without 3: costs and `mip_gap`).
  Its rows have no episode index; I aligned them by matching `J_relaxed` to `J_oracle_cents` of the reference cache.
- `outputs/hazard_lab/play/{base_s,w50as_s,ends_all_s,...}_small_444.pkl`, `{base_f,w50as_f}_full_444.pkl`
  (weekly cost items, lots, energy, stock, sent of the current planner; `w50as` is the hazard model's setting).
- `hub/refcache/references/v0.1.2-39ec701c95ac/{small,full}/*/entropy-444/` (naive, clairvoyant, harm per episode).
  Harm levels of root 444 Small from the borders in FINDINGS (1 648 / 2 201 / 2 811 bn): 18/12/5/5 on 40 episodes,
  the same split `plan_stats_small.md` prints; my scoring reproduces 0.95 (plan), 0.8975 (`base_s` x64), 0.9080
  (`base_s` eps 0-23, P7 has 0.9079). Full 444 has no borders on record: Full numbers are ratios of sums.
- `outputs/regime_lab/{descend,hull,switch}/small_444_{0..7}_*.pkl` (older executed descents, cost per episode).
- The instance itself (`task_generator('small'|'full')`): grid priorities, shares, storages, salvage.
- Code: package `oracle/lp.py`, `dynamics/sim.py`, `dynamics/production.py`; `lab/anastasiia/stats_lab/plan_stats.py`.

Scale: 0.01 RSS = 10.1 bn USD per episode on Small 222, 34.5 bn on Full 222 (sum of level weight x mean room).

---

## 1. The distance from the model to 1.0, decomposed

### 1a. What the formal records say (measured, `anastasiia_plan_hazard`)

| | Small 222 x256 | Full 222 x128 |
| --- | --- | --- |
| RSS, levels 1-4 | 0.899: 0.897 / 0.897 / 0.901 / 0.928 | 0.902: 0.907 / 0.906 / 0.893 / 0.855 |
| gap to clairvoyant, bn/ep, levels 1-4 | 107 / 107 / 92 / 64 (mean 102, median 70) | 311 / 323 / 413 / 480 (mean 334, median 246) |
| gap as % of the level's room | 10.3 / 10.3 / 9.9 / 7.2 | 9.3 / 9.5 / 10.7 / 14.5 |
| share of the weighted gap | 52 / 31 / 13.5 / 3 % | 46 / 29 / 18 / 7 % |
| per-episode RSS: 5 % / 25 % / median / 75 % / max | 0.673 / 0.852 / 0.929 / 0.963 / 0.994 | 0.780 / 0.866 / 0.921 / 0.949 / 0.978 |
| worst 5 % / 10 % / 20 % / 50 % of episodes hold (RSS) | 0.019 / 0.032 / 0.051 / 0.084 of 0.101 | 0.016 / 0.028 / 0.045 / 0.074 of 0.098 |
| episodes cheaper than the reference | 0 (closest 2.4 bn) | 0 (closest 74 bn) |
| corr(episode RSS, room) / (episode RSS, naive cost) | +0.13 / -0.22 | +0.21 / -0.12 |
| terciles of room: RSS low / mid / high | 0.888 / 0.883 / 0.915 | 0.872 / 0.914 / 0.912 |
| per-episode best of the 5 recorded models | 0.9008 (+0.002) | 0.9033 (+0.002) |
| corr of per-episode gap, hazard vs rules_v2 / hybrid_hub / hull3 | 0.91 / 0.94 / 0.995 | 0.90 / 0.92 / 0.994 |

Reading:
- **Level 1 is not a weak level, it is a heavy level.** The gap per episode is the same in levels 1-3 on Small and
  level 1 is the *best* level on Full. Level 1 holds half of the gap because it holds half of the weight.
- **The median episode already scores 0.93 / 0.92.** The mean is pulled down by a tail: the worst tenth of the
  episodes holds a third of the distance.
- **The tail is a property of the episode, not of the model**: the per-episode gap of the rules agent correlates
  0.90-0.91 with the planner's. An oracle that picked the best of our five models per episode gains 0.002: there is
  no portfolio pool.
- The reference is an exact LP optimum (IPM, `oracle/lp.py:83`), so it has no noise of its own; what is noisy is the
  set (+-0.016 on 256 Small episodes).

### 1b. The ladder on root 444 (measured unless marked)

Small, board weights:

| rung | eps 0-7 | eps 0-23 | eps 0-39 | what it is |
| --- | --- | --- | --- | --- |
| `base_s` (hull3 settings) | 0.8825 | 0.9080 | 0.9067 | played |
| hazard settings (`w50as_s`) | 0.8863 | 0.9137 | 0.9127 | played |
| + true ends of every running event in the window (`ends_all_s`) | 0.8971 | 0.9192 | 0.9158 | oracle, played |
| descent with the whole future, executable | 0.9230 | 0.9402 | - | P7 (`BOARD.md`), quoted |
| base-first MILP, incumbent (its own cost) | 0.9343 | 0.9518 | 0.9503 | `J_plan` |
| **base-first MILP, dual bound** | **0.9499** | **0.9628** | **0.9625** | `J_plan x (1 - mip_gap)`, new here |
| clairvoyant LP | 1 | 1 | 1 | |

Full, ratio of sums (levels not weighted):

| rung | 12 episodes with MILP gap < 3 % | 15 episodes the model has |
| --- | --- | --- |
| hazard settings (`w50as_f`) | 0.9135 | 0.9021 |
| base-first MILP incumbent (300 s) | 0.9569 | 0.8994 |
| **base-first MILP dual bound** | **0.9774** | 0.9772 |

(16 episodes with gap < 3 %: incumbent 0.9544, bound 0.9745. All 19: 0.9061 and 0.9738.)

**The dual bound is a proven ceiling for any agent on these episodes**, with or without the future: every simulator
trajectory satisfies the oracle LP's rows with the same cost (`oracle/lp.py:13`, the regime_lab check to 2e-7), and
every grid of both networks is `base_first` (checked on the instance), so every trajectory is feasible for the MILP
of `plan_stats.py:31-76` (the oracle LP plus one binary per grid and week: shed = 0 or fab energy = 0).

So the distance of the hazard model to 1.0 on Small 444 eps 0-23 (0.086) is:

| pool | size | status |
| --- | --- | --- |
| ends of running events inside the window | 0.0055 left (0.011 from `base_s`) | measured |
| the rest of model -> executable descent | 0.021 | measured as a difference; its split is **not** measured (see 3c) |
| descent -> MILP incumbent: the simulator's other automatic rules, or the descent's locality | 0.012 | measured as a difference |
| MILP incumbent -> MILP bound: solver slack | 0.011 | measured; which side is right is unknown |
| **MILP bound -> clairvoyant: proven unreachable** | **0.037** | measured (0.050 on eps 0-7, 0.0375 on eps 0-39) |

On Full (12 well-solved episodes; model 0.9135): model -> incumbent 0.044 (154 bn/ep), incumbent -> bound 0.021
(73 bn), **bound -> clairvoyant 0.023 (80 bn) proven unreachable**. The Full descent is not on this ladder because it
was never measured from the current trajectory with hull rounds (FINDINGS line 768).

Carried to the formal sets (**inference**, proportions of root 444): of 0.10 to the reference, about 0.04 on Small and
0.025-0.045 on Full is the relaxation, about 0.01-0.02 is undecided solver slack, and 0.04-0.05 is in principle
reachable with the whole future, of which the descent has demonstrated 0.027-0.035 on Small.

### 1c. The relaxation tax is concentrated, and it is what makes "bad episodes" bad (measured, Small 444 x40)

- MILP incumbent minus clairvoyant per episode, as a share of room: 25 % quantile 0.1 %, **median 1.3 %**, 75 % 8.3 %,
  max 18.9 %. The top 4 episodes hold 47 % of it, the top 8 hold 74 %, the top 20 hold 98 %.
- In the 24 episodes HiGHS closed to 0.2 % (58 % of the room) the base-first plan scores **0.9865**; in the 16 it
  did not close, 0.898 (bound 0.926). The hard-to-solve episodes and the taxed episodes are the same ones.
- **corr(model's gap to clairvoyant, base-first tax) = 0.96 across episodes.** corr(model's gap to the MILP plan,
  tax) = 0.24.
- By level, share of room: tax 6.3 / 4.5 / 1.6 / 1.1 %; model -> MILP incumbent 3.6 / 4.5 / 2.4 / 3.2 %.
  The calm level looks worst only because the tax is largest there (**inference** on why: in calm episodes exits and
  wafers are open, so fab power is the binding constraint and the relaxation's trick is worth most).
- **The reachable part is a flat tax, not a tail**: model minus MILP incumbent per episode has quantiles
  12 / 24 / 40 / 49 / 74 bn (10 / 25 / 50 / 75 / 90 %), mean 40; the worst 8 of 40 hold 39 %. Compare the strike
  oracle, whose gain is 65 % in 8 episodes of 64 (`hazard_lab/README.md`).
- Full, per episode (19): the bound's episode RSS is 0.93-1.00 in 17 of 19; the tax is in episodes 2 (proven 14 % of
  room), 16 and 18.

Consequences: "the worst episodes are already at the ceiling" (FINDINGS 273) is confirmed and generalised - their
gap is the proven tax. Lists of worst episodes, the level-1 story and a CVaR aimed at the tail all point at money
nobody can take. The reachable money is about 40 bn in *every* Small episode.

### 1d. The reachable gap by cost item and by time (measured, Small 444 x40, hazard settings vs the MILP incumbent)

Total 40.4 bn/ep = unmet demand 26.5 + shed 10.7 + other 3.2.

| | weeks 1-13 | 14-26 | 27-39 | 40-52 |
| --- | --- | --- | --- | --- |
| shed, model minus plan, bn | -33.1 | +21.3 | +20.9 | +1.6 |
| unmet demand, model minus plan, bn | -2.9 | -5.7 | +10.7 | +24.3 |
| lots started, model / plan, thousand (all six fabs) | 3 552 / 1 999 | 3 051 / 3 145 | 2 657 / 2 988 | 331 / 65 |
| fab energy, model / plan, GWh | 4 703 / 2 701 | 4 121 / 4 242 | 3 608 / 4 037 | 437 / 63 |
| `chip_le` served at `sink_us`, thousand per week | 216.7 / 215.5 | 187.4 / 185.9 | 138.8 / 150.8 | 132.5 / 159.4 |

(The two rows of lots and fab energy are cut at weeks 38 / 39, not 39 / 40: a lot started after week 38 cannot sell.)

- Unmet demand is `chip_le` (25.7 of 26.5), and it is `sink_us` (23.7) and `sink_eu` (3.8). Shed is JP (7.1 of 10.7).
- The plan sells 663 thousand more `chip_le` in weeks 27-52 from only 298 thousand more `chip_le` lots in weeks
  14-38: about half is production timing, half is where the chips are.
- In weeks 40-52 the exits to the US are **not** full: open first-edge capacity osat -> `sink_us` 3 181 thousand,
  model 1 632 (51 %), plan 1 921 (60 %). The model ends the episode with 502 thousand finished `chip_le` on the
  packaging plants against the plan's 219 thousand. But chips "on hand at week start, not dispatched while the US exit
  has slack" are only 47 thousand (2.4 bn) in weeks 27-52: the model ships what it can; its stock sits where the US
  exit is closed (27-38 % of plant-weeks), and plants with an open exit are empty.
- Checked and refuted: wafers are not what limits late lots (the model keeps 1.7 M wafers at `mat_jp_wafer` and
  266-299 thousand at the KR fab against a weekly capacity of 305 thousand).
- The early-shed shift (-33 then +42) does not explain the gap per episode (corr 0.11 with the total) and is likely
  the MILP holding fuel on the grid unburned (FINDINGS 309-311 says the same of v2): **not a pattern to copy**.
- Against the clairvoyant the whole gap is chips: 85.9 of 90.6 bn (`chip_le` 76.2), shed 2.0.
- Full by item against the base-first plan for the current model: **not computed** (the Full file holds costs only).

---

## 2. The relaxation gap: what exactly the reference drops

`oracle/lp.py` with `planning_rules=False` against `sim.py` step 7 and `production.py`:

| # | simulator rule | where | what the LP has instead | can an action pattern emulate the LP? |
| --- | --- | --- | --- | --- |
| 1 | base load first: `y = min(y_bar, g_av)`, fabs get `g_av - y` | `production.py:25-27`, `sim.py:350` | `y + sum E <= sum G` and `y + ysh = y_bar` (`lp.py:521-522, 636-640, 700-708`): a fab may draw while base is shed | **no**. Fab energy = max(0, top - sum of shortfalls); nothing the agent sends changes `G_bar` or `y_bar` |
| 2 | every segment burns `av_k x load`, the fuel-free one too; fuel on the grid burns whenever there is load | `sim.py:355-361` | `G_gk` free in [0, share x G_bar] (`lp.py:675-676, 695`): fuel may wait on the grid unburned, and the free segment may run alone | yes, one step upstream: fuel waits at the terminal (valve lag 0) or in a strait queue (no cap, `lp.py:580`) |
| 3 | fab starts all wafers it has power for; grid energy split pro rata to requests | `sim.py:322-326, 354`, `production.py:27` | `p <= cap`, `e p <= R E`, `p` free (`lp.py:609-640`) | yes: wafers wait at the source; the split is steered by which fab holds wafers |
| 4 | packaging plant packs all raw chips up to throughput, pro rata between products | `production.py:41-49` | `xi` free (`lp.py:650-658`) | mostly: raw chips wait at the fab, limited by the fab's small raw store |
| 5 | containers leave a strait oldest first | `chokepoint.py` | lane-indexed queue with free release (`lp.py:504-510, 551-553`) | only by holding at the origin |
| 6 | gas ration on last week's stock | `sim.py:337-340` | the same row (`lp.py:696-699`) | not dropped |
| 7 | clipping, fleet slack, supply lift, disposal, serving | `clip.py`, `sim.py:316-319, 395-412` | the feasible sets they project onto | not a relaxation once capacities are known |

Sizes:
- Rule 1 alone: **at least 0.037 (Small 444 x40) and 0.023-0.026 (Full 444) by the dual bound; at most 0.050 and
  0.046 if the incumbents are optimal.** Not 0.06-0.08.
- Rules 2-5 together: **at most 0.012 on Small** (descent -> MILP incumbent on eps 0-23 and 0-7). In 5 of the 8
  episodes with a saved descent the executed plan is within 0.3-12 bn of the MILP incumbent; in the three others
  (episodes 1, 3, 7: 25, 37 and 64 bn) the MILP's own slack is 12, 85 and 12 bn. FINDINGS 432-437 states the mechanism: each is undone one step
  upstream. On Full: not measured.
- Which fabs carry rule 1 on Small (extra lots of the relaxed plan, valued): JP memory 29.1 bn (49 %), KR memory
  17.0 (29 %), EU mature 7.3 and EU leading 4.8 (20 %), TW mature 1.6, TW leading 0. Cost side: the relaxed plan
  sheds 8.7 bn more to do it.

The specific questions:
- **Any grid where the rule does not bind?** No grid is exempt: all 4 + 9 grids are `base_first`, no fab is off-grid
  or free of energy (instance, checked). It binds weakly where a fuel is smaller than the top sliver: crude at TW
  (2 % against a 7.0 % sliver) and at KR (2 % against 3.5 %). `grid_us` on Full is fed by pipe (FINDINGS 635).
- **Wafers steering demand and burn.** With no wafers the grid's load factor is `y_bar / g_av`, so every segment
  burns less: this is the only way to keep fuel on the grid unburned, and it saves only the sliver (7 % of a TW week,
  1 % of JP, 0.4 % of EU). Worth it only for lots that would not sell. The model starts 1.55 M more lots than the
  plan in weeks 1-13 (2.0 TWh, about half of it fuel): a ceiling of about 4-8 bn/ep, the same thing as the measured
  throttle (11 bn, `heur_lab2/reports/throttle.md`). **Inference** for the ceiling.
- **Grids with several fabs.** A partly short week still powers the dearest fab fully if the cheaper one holds no
  wafers (EU: a shortfall up to 162 of 211 GWh leaves the leading fab whole). At most about 0.5 bn per such week.
  Whether the regime cell already does this: **not checked**.
- **Storage.** Gas terminals hold 4.0 weeks of full burn at TW, KR, JP, EU on Small, the grid another 4, strait
  queues are uncapped. **On Full `term_eu` holds 7 288 = 0.8 week of EU burn** (Small: 35 346; assuming EU's
  generation is the same on Full, as its shares and reference buffer are): on Full the "wait at the terminal"
  emulation of rule 2 is storage-bound at EU. Not seen in FINDINGS; whether the agents already hit it: not checked.
- **The two-week ration dance.** A grid whose gas stock was 0 last week burns nothing this week whatever arrives, so
  the threshold stock (0.94-1.2 weeks of burn) can be put on the grid in one "dark" week and is not wasted. With the
  whole future it costs timing only; under uncertainty it costs a dark week per restart. **Inference** from the code.
- **Order of segments.** There is none (pro rata); in a surplus week fuel burns instead of the free segment unless
  the fuel is not on the grid. Part of the throttle number above.
- **Terminal credit.** Salvage is 500-2 000 USD per fuel unit against 4.1 M per shed GWh and 2-18 USD per chip
  against 50 thousand; the model's whole credit is 0.6 bn/ep. No pool.

Verdict: rule 1 is the relaxation; no legal pattern approximates it. But the amount proven lost to it is half of
what the team writes off.

---

## 3. Is 0.92 the executable ceiling?

a. **Small.** 0.92 is the number of the worst subset. On eps 0-23 the executable descent is at 0.940 and the proven
   bound at 0.963; on eps 0-7, 0.923 and 0.950. The true executable optimum is in [0.940, 0.963] on the 24-episode
   set. The 0.95 of the MILP is not evidence of a much higher optimum: it is the incumbent of a problem that still
   relaxes rules 2-5, and the descent already sits within 0.012 of it. Upside over the descent: 0.01 likely,
   0.023 at most. **The belief holds to within 0.02 on Small.**

b. **Full. The belief is unsupported, and the evidence points higher.**
   - The "0.90-0.93" came from descents started at `hybrid_chiplp` (0.835), 5 rounds, no hull rounds, 8 episodes;
     FINDINGS 766-769 itself says it must be re-measured.
   - Where HiGHS closed to 3 %, the base-first incumbent is 0.954-0.957 and the bound 0.975-0.977; the model is at
     0.9135 on the same 12 episodes. Model -> incumbent is 0.044 on Full against 0.037 on Small.
   - In 4 of 15 episodes the causal model is **cheaper than the 300-second MILP incumbent that knows the future**
     (by 27, 64, 500 and 1 421 bn; their MILP gaps are 2-22 %). Every Full "ceiling" so far is a statement about the
     solver.
   - If rules 2-5 cost on Full what they cost on Small (at most 0.012), the executable ceiling on Full is about
     0.94-0.945 on well-behaved episodes (**inference**). Against it: `term_eu` storage (section 2).

c. **Is model -> descent information?** On Small 444 eps 0-23: `base_s` 0.9080, with the true ends of all running
   events 0.9192, descent 0.9402. Onsets of new events were worth +0.003 on Small for an older planner (FINDINGS
   774). That leaves **about 0.016-0.018 that window information does not explain** (inference by subtraction, from
   runs of different planner versions). A longer window is not it: 52 weeks equals 26 with end prices on Small
   (`plan_lab/README.md:361-375`, older planner). `BOARD.md` P7 already calls the 0.026 / 0.009 split unconfirmed.
   What remains is whole-episode coordination: the descent iterates the regime reading to a fixed point with hull
   rounds and search; the agent solves once or twice a week from a reference that changes under it.

d. **What a better global optimiser of the full-future problem needs.**
   - A start: any played trajectory is MILP-feasible, so the model's own trajectory lifted into the LP columns is an
     incumbent. That alone removes the garbage incumbents on Full.
   - Locality: few cells are contested (103 of 115 hull cells ask for zero, SPEC P17). Fix the binaries where the
     descent, the hull and the LP agree and solve the rest exactly (local branching, k flips): a provably optimal
     neighbourhood instead of hill climbing, and a bound that tightens at the same time.
   - Rules 2-5 need not be modelled exactly: they cost at most 0.012 and the descent repairs them.
   - It stays an offline tool. Conversion of ceiling into play has been 13-55 % (0.0195 -> +0.0025 for same-state
     plan comparison; 0.009 -> +0.005 for ends). Raising the Small ceiling by 0.01 buys a causal agent 0.002-0.005.
     On Full the stake is larger because the ceiling itself is unknown.

---

## 4. Directions, ranked by the pool they open

### D1. Measure the Full ceiling properly: MIP start from the played trajectory, then hull descent
- **Pool:** on Full, model -> incumbent 0.044 and -> bound 0.064 (12 episodes); the final is scored on Full.
- **Mechanism:** section 3b and 3d. Lift the hazard trajectory into the MILP as a start, 300-900 s per episode,
  Full 444 eps 0-15; then the P7 descent with hull rounds from the same trajectory.
- **Evidence:** incumbent 0.957 / bound 0.977 against 0.9135; 4 of 15 incumbents worse than the causal model.
- **Not dead:** P7 was run on Small only; FINDINGS 768 asks for it; P17(a) was kept "as a bound".
- **Cheapest decisive experiment:** offline, 16 episodes, one machine-evening (after the formal run).
- **Closes it:** descent with hull <= model + 0.015 on these episodes and re-solved incumbents <= 0.93.
  Opens it: descent >= 0.94 - then Full has 0.03+ of teacher signal the Small-derived plan does not expect.

### D2. Split model -> descent into information and planner with one oracle run
- **Pool:** 0.016-0.018 on Small that ends and onsets do not explain; Full unknown.
- **Mechanism:** play the current planner with the true future of every field in its window (new onsets included)
  on Small 444 eps 0-23 and Full 444 eps 0-7; compare with `ends_all` (0.9192) and with the descent (0.9402).
- **Not dead:** `truth_events` gives ends only; the "all fields" run exists for `plan_hull` on other episodes.
- **Closes it:** truth-in-window >= descent - 0.005: the gap is information, only P3 can touch it.
  Opens it: truth-in-window <= 0.925: 0.015 is reachable with no forecast at all, by more regime iteration per
  week where the budget allows (the Small cell is 0.3-0.4 s) and by P2.

### D3. Put every diagnostic on the proven bound, not on the clairvoyant
- **Pool:** none directly; it redirects P3's CVaR, the "level 1" plan and the worst-episode lists away from 0.037
  (Small) that is proven unreachable and that correlates 0.96 with the model's gap.
- **Experiment:** offline. Root 444 is done here. Root 111 x64 has the MILP solutions and gaps on disk
  (`outputs/regime_lab/basefirst/small_111_*_tl60.pkl`); it needs 64 LP builds to price them (minutes).
- **Closes it:** on root 111 the tax is not concentrated (top fifth under 50 %) or its correlation with the model's
  gap is under 0.5.

### D4. The late `chip_le` at `sink_us`: 35 of the 40 bn of reachable gap, and nobody owns chips
- **Pool:** 26.5 bn/ep of unmet demand against the MILP plan on Small (0.026); 16.7 of 29.5 against the executable
  teacher on 4 episodes (`cycle_lab/README.md`). All open directions are about fuel weeks and the window's end.
- **Mechanism (hypothesis):** the model's finished chips sit at plants whose US exit is closed while plants with an
  open exit run empty; the decision is the fab -> plant split 3-5 weeks earlier and the fab-weeks 14 weeks earlier.
- **Not dead:** block decomposition is dead; this is a diagnosis and one objective term, the form P11 measured +.
- **Cheapest decisive experiment:** offline on the pickles above: for weeks 20-38, compare the model's fab -> plant
  split with the plan's, conditioned on the exit state *observable in the dispatch week* (bans persist 93-95 %).
- **Closes it:** over 70 % of the plan's advantage needs exit changes that happen after the dispatch week.

### D5. A Full-only physics audit before trusting Small-derived rules
- **Pool:** unknown. `term_eu` gas storage is 0.8 week on Full against 4.0 on Small; at CN the gate fuels are 1-5 %
  of generation behind a sliver of 0.22 %; FINDINGS 652-660 counts 182 near-full weeks per episode that nobody
  closes.
- **Experiment:** offline from the Full play pickles (`base_f`, `w50as_f` hold stock, segment, energy by week):
  full weeks lost at EU and CN by cause (terminal full, valve, ration, second gate fuel).
- **Closes it:** under 10 bn/ep attributable to terminal storage at EU.

Not proposed, and why: a portfolio of agents (0.002 with hindsight); terminal credit (0.6 bn/ep); longer window
(measured null on Small); anything that powers a fab on a shedding grid (no legal pattern, section 2).
