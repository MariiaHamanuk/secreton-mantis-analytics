# Small: the gap of the rule agent to the best plan, foresight or control (6 October)

Root 444, episodes 0..39; scripts and logs of that day in outputs/heur3/small_gap/ (local, not in git)

All numbers below come from saved arrays (root 444, episodes 0..39, no simulation). Scripts and logs are in `$MAIN/outputs/heur3/small_gap/` (`s1`..`s6*.py`, matching `*.log`, `common.py`). "measured" means the script produces it. "inferred" means my reading of it.

## Summary

- **The gap does not depend on how much the network changes.** Across the 40 episodes, the per-episode gap has essentially zero correlation with my network-change index. The extrapolated gap at a static network is about 111 of 113.6 bn. This suggests most of the gap is not foresight of network events, but see the limits below.
- **The plan's own cost is not executable.** The plan replayed blindly in the same scenario is 87.2 bn worse than its own cost [90% CI 78, 96]. That is model-versus-simulator mismatch only, because the scenario is identical. The replay also beats v2 in only 14 of 40 episodes, by 26.4 bn [15, 38] on average.
- **Chips are about waste, not market allocation.** The plan sells 0.92 M more chip_le, mostly in weeks 40-52. v2 strands or disposes about 3.4 M chip_le along the chain against the plan's 0.4 M.
- **Shed is about fuel stock management plus lost free energy.** The plan delivers about the same fuel as v2 and burns about 7 TWh more of it. That is where about 30 of the 54 bn comes from; the stock-path mechanism is not reconciled with the ration formula.
- **Verdict.** I cannot tell whether more rules add 0.05 RSS (about 55 bn). Plausible rule-recoverable gap is roughly 25-70 bn (0.02-0.065 RSS), with a wide uncertainty band. See table 6.

## 1. When the gap arises (measured: `s1_periods.py`, `s1.log`)

USD bn per episode, plan minus v2, mean over 40.

| period | chip_le | chip_mat | shed | sum |
|---|---|---|---|---|
| 1-13 | -6.4 | -2.7 | -5.7 | -14.8 |
| 14-39 | 23.7 | 5.3 | 51.7 | 80.7 |
| 40-52 | 29.2 | 3.4 | 7.7 | 40.3 |
| total | 46.5 | 6.0 | 53.8 | 106.2 |

- The sum is 106.2, plus 7.4 "other", which equals the 113.6 total.
- In weeks 1-13 v2 beats the plan on both chips and shed. There the gap is negative (-14.8).
- Weeks 14-52 carry 121 bn.
- Shed gap in weeks 14-39 by grid (TW, KR, JP, EU): 7.5, 12.3, 3.1, 28.8. EU in weeks 1-13 is -16.9 (v2 better), so EU is a timing swap.
- chip_le sales (M units): weeks 1-13 plan 4.29 vs v2 4.42; weeks 14-39 6.51 vs 6.04; weeks 40-52 3.14 vs 2.56.
- About 63% of the chip_le shortage gap is in weeks 40-52.

## 2. Static against changing networks (measured: `s0_marks.py`, `s2_static.py`, `s2b_stats.py`, `s2c_prechange.py`, `s2d_boot.py`)

**Change index.** I used the true marks: edge capacity, strait openness, prohibitions, source supply, fab restoration and grid output.
- Per slot: relative effective capacity per week (first-edge capacity, lane straits, bans), compared with its own week-1 value.
- The index is the mean of the shares of slot-weeks differing by more than 0.1 in the fuel, wafer and chip groups, plus source supply and fab restoration.
- Only 8 of 40 episodes have index below 0.02, and none are strictly static.
- In every episode the week-1 network is already heavily disrupted (mean relative capacity about 0.5), and week-1 values mostly persist.
- Index quantiles (0, 25%, 50%, 75%, 100%): 0.005, 0.023, 0.035, 0.060, 0.162.

**Gap by group (bn per episode):**

| group | n | gap | chips | shed | other | replay minus plan |
|---|---|---|---|---|---|---|
| index < 0.02 | 8 | 98.6 | 59.7 | 31.5 | 7.4 | 91.4 |
| index ≥ 0.02 | 32 | 117.4 | 50.7 | 59.3 | 7.4 | 86.1 |
| index < 0.05 | 24 | 114.1 | 57.3 | 49.7 | 7.1 | 88.1 |
| index ≥ 0.05 | 16 | 112.9 | 45.2 | 59.9 | 7.8 | 85.8 |
| first big change ≥ week 36 or none | 7 | 119.7 | 68.7 | 43.5 | 7.5 | 97.9 |
| first big change < week 36 | 33 | 112.3 | 49.0 | 55.9 | 7.4 | 84.9 |
| lowest tercile of index | 14 | 118.1 | | | | |
| highest tercile of index | 14 | 119.3 | | | | |

**Regression of gap on index** (a is the intercept, b the effect per +0.1 of index, bn):

| response | a | b per +0.1 | r | p |
|---|---|---|---|---|
| gap | 111.0 [91, 132] | 5.8 | 0.04 | 0.81 |
| chips | 57.4 | -10.8 | -0.09 | 0.58 |
| shed | 46.8 | 15.3 | 0.13 | 0.42 |
| replay minus plan | 83.3 | 8.6 | 0.08 | 0.64 |

- Other change measures give the same result: share of changed fuel slots (shed r 0.28, p 0.08; Spearman 0.40) and 4-week persistence error (gap Spearman 0.17, n.s.).
- The predicted rise in gap from index 0.005 to the mean index of 0.045 is 2.5 bn, 90% CI [-10.7, +16.4].
- Gap SD across episodes is 49, so a group mean of 7-8 episodes has a standard error of about 18 bn.

**What this implies:**
- measured: the gap does not rise with the amount of network change in these 40 episodes, and the replay-minus-plan loss does not either (as it should not; it is a mismatch).
- inferred: the share of the gap that needs foresight of network events is small, about 2.5 bn point estimate. Taking the upper end of the 90% interval gives at most 15-20% of the gap.
- A caveat applies to the "the network barely changes, so static means perfect forecast" reading. Even in episodes with no network change, the plan still knows the future of demand, the initial pipeline schedule and, in episodes with a later change, that it will happen.
- Also, in the high-change group a rule that reads week-t information still gets the gap.
- Firmness: weak, because n=40 and only 7-8 near-static episodes. A foresight share of 0-20% is consistent with the data. A 30% share is not excluded, but it would need an effect I do not see.
- Within-episode check (`s2c_prechange.py`, 17 episodes with a first change at week 12 or later): shed gap per week is 1.38 before the change against 0.95 after. Chip gap per week, shifted by 8 weeks for production lag, is 0.09 before against 1.89 after. Some chip loss may follow changes, but the sample is small and noisy.

## 3. Chips (measured: `s3_chips.py`, `s3b_balance.py`, `s3c_jp.py`, `s3d_plants.py`, `s3g_chain.py`, `s3h_disp.py`, `s3i_closed.py`)

**Market allocation does not carry value.** The price is about 50.4k per chip_le in every market. The plan's JP shortfall (-0.40 M) is offset by US +0.86 M and EU +0.46 M; this sums to +0.92 M of extra total chip_le sales (46.4 bn at 50.4k). v2 sells 13.02 M and the plan 13.94 M. The market shift is tie-breaking among equal prices (freight and tariff). It is not a causal 66 bn gain and 20 bn loss.

**Where chip_le goes, M units per episode (v2 | plan):**

| item | v2 | plan |
|---|---|---|
| raw produced at chip_le fabs | 12.81 | 10.93 |
| raw shipped out of fabs | 10.85 | 11.33 |
| disposed at fabs | 1.89 | 0.13 |
| left at fabs at end | 0.62 | 0.01 |
| packaged left at plants at end | 0.75 | 0.22 |
| disposed at plants | 0.10 | 0 |
| sold | 13.02 | 13.94 |

- The plan makes 1.89 M fewer chips but ships 0.48 M more raw and wastes about 3.0 M fewer.
- Plan "disposed" is a residual from mass balance (plan production minus shipments minus change in stock), so small plan values are noisy.

**chip_mat:** v2 produces 3.70 M, ships 2.91 M, disposes 0.82 M and sells 4.62 M. The plan produces 3.26 M, ships 3.29 M, disposes 0.16 M and sells 5.19 M.

**Flows by plant (K units per episode):**
- JP fab to osat_my: v2 1581, plan 2071 (+490).
- EU mature to osat_my: 1107 against 1376.
- TW and KR fabs ship about equal amounts.
- From the plants, the plan sends more to US (osat_my +461, osat_kr +216, osat_tw +190) and to EU (osat_my +299, osat_kr +158), and less to JP (osat_my -281).
- The plan's larger extra sales are in weeks 40-52. It sells 3.14 M against 2.56 M.

**JP fab (K units a week).**
- v2 ships 58-63 in weeks 1-8 while the route capacity is about 170 and its stock rises from 285 to 374 (storage 518). It disposes 64 a week in weeks 5-8.
- The plan ships 77-90 and does not dispose.
- v2 keeps 150-250 in stock at the JP fab throughout; the plan's stock goes to about 0.
- KR is similar in kind. The plan's fab stock rises to about 850 early on because it overproduces less.

**Timing of v2 fab raw disposal:**
- JP: 23 bn of its value in weeks 1-13.
- KR: 14 bn in weeks 1-13, 20 bn in weeks 14-39 and 6 bn in weeks 40-52.
- Total fab raw disposal is 104 bn at sale price, an overestimate because packaging and market routes also limit sales.
- 63% of disposed units (1.7 M of 2.7 M) occur when the fab's outbound capacity was at least 25% of nominal, 34% when below 25%, and 4% when zero.
- `asked` equals `sent`, so these are v2's own choices, not clipping.

**Plants and end stocks:**
- v2's plant stock averages 500-650K packaged chip_le.
- In many episodes this reflects closed outbound routes: at osat_my, 0.32 of 0.40 M end stock sits behind routes under 25% open.
- In episodes 6 and 27, v2's plants are starved of raw (stock equals what it sends each week) and the plan ships 3-5 times more.

**Classification:** mostly production location and timing of shipments (what goes where and when), plus end-of-horizon stranding. Market allocation is value-neutral. The information needed (stock levels, storage, open routes) is visible every week. Not every part is foresight-free. Chips sitting behind routes that close later need the future.

## 4. Shed (measured: `s4_shed.py`, `s4b_energy.py`, `s4c_energy2.py`, `s4d_eu.py`, `s4e_eu_burn.py`)

**Fuel into each grid (v2 vs plan; arrivals by balance; nuclear fuel arrivals not modelled, ignore those rows).**

| grid | fuel | burned v2 (K) | burned plan (K) | delivered v2 (K) | delivered plan (K) |
|---|---|---|---|---|---|
| TW | lng | 78.6 | 78.7 | 76.5 | 75.3 |
| KR | lng | 104.7 | 106.5 | 102.3 | 102.6 |
| JP | lng | 166.8 | 167.2 | 159.6 | 157.0 |
| JP | crude | 23.6 | 25.9 | 23.7 | 25.9 |
| EU | lng | 245.5 | 249.3 | 233.3 | 232.6 |

- The plan delivers about the same fuel as v2. It does not deliver more, and its lng and crude deliveries are 1-3% lower in most grids.
- It burns more because it leaves less stock at the end of the episode. Grid stocks at the end, lng (K): v2 1.3, 1.6, 3.6, 5.4 (TW, KR, JP, EU, sum 11.9) against plan 0.02, 0, 0.6, 0.9 (sum 1.6).
- v2 also strands lng at terminals (8.0 K) and 5.1 K at straits; the plan ends with about 0 at terminals.

**Energy accounting, TWh per episode (the shed gap is 13.0 TWh = 53.8 bn):**

| item | TWh | bn |
|---|---|---|
| more fuel burned (plan minus v2) | +7.2 | about 30 |
| v2 free segment lost (plan loses none) | 2.8 | 11.5 |
| fab energy diverted | 3.1 | 12.7 |

- "Fab energy diverted" (v2 14.1 TWh, plan 11.0 TWh) is accounting only. Across episodes shed gap against extra fab energy has R² 0.02, so I do not attribute shed to it causally (`s5c_energy_reg.py`).

**By period (plan minus v2 base load served, TWh):** weeks 1-13 -1.38; weeks 14-39 +12.54; weeks 40-52 +1.87.
- EU is the swing grid: -4.09, +6.98, +1.15 TWh in the three periods, with no change in EU's lng route capacity (0.84 throughout).
- The EU lng route openness change is unrelated to the plan's EU advantage (Spearman 0.09, p 0.59).

**Stock path:**
- The plan keeps grid lng stock far above v2's in weeks 5-45 and ends near zero.
  - EU: 7-9 K against 5.2 K.
  - KR early: 3.7 K against 2.2 K.
- It burns less in weeks 1-8 (EU 8.1 vs 8.7 K a week; JP 4.65 vs 5.33; KR 1.95 vs 2.6; TW 1.4 vs 2.04).
- It burns 5-15% more in the middle of the episode.
- Weeks served at 99.9% or more of base load: EU 28% (plan) against 22% (v2); JP 25% against 22%; KR and TW about equal.
- inferred: this is storing early and then dumping, so the plan spends more weeks with fuel available when the free segment makes the grid offer at or above base load.
- Unreconciled: with ration equal to closing stock over threshold, the plan's higher stocks (ration about 0.7-0.85) should burn more than it does. I derive plan burn of 4.4-4.6 K in the middle weeks. I found no explanation (plan may be throttling in weeks without shed). Treat the plan's burn path as unexplained, not as a rule.

**Verdict on item 4:** the advantage comes from fuel stock timing (about 55%), not losing the free segment (about 21%), and a mostly non-causal fab-energy accounting term (about 24%). No evidence that the plan delivers more fuel.

## 5. Six episodes (measured: `s5_eps.py`, `s5b_stuck.py`)

Gap in bn, split chips / shed, and RSS where known.

- **Episode 6** (gap 257 = chips 117 + shed 139, RSS 0.73, stratum 1). The plan starts more lots at every fab (KR 9.06 M against 8.40 M) and shuns JP: JP shortage 0.69 against 0.25 for v2, a market swap. Plan fab energy is higher in every grid (all 6 fabs; for example 5644 against 4202 at TW leading).
  - In weeks 1-13 the shed gap is -63 (v2 better); in weeks 14-39 it is +151. Outbound capacity is 0.84-1.0 and does not change, so this is control quality.
  - A rule can close part of it: the plan runs the KR fab flat out in weeks 1-40 while v2 oscillates, which is a fab-start and feed-timing defect.
  - The 151 bn in weeks 14-39 is mostly fuel timing I cannot attribute to a rule.
- **Episode 27** (gap 239 = chips 123 + shed 111, RSS 0.70). Same pattern: the plan starts 9.06 M lots at KR against 7.75 M. v2 loses to the plan by 194 bn in US chip_le and the plan loses 98 bn in JP.
  - The shed gap is -72 in weeks 1-13 and +170 in weeks 14-39, mostly KR (69) and JP (43).
  - There are 12 events with onset after week 1, so some later foresight exists.
  - v2's KR and JP fabs are not fed enough in the middle; probably rule-recoverable timing and feed, but the size is not identified.
- **Episode 8** (gap 192 = chips 107 + shed 80, RSS 0.81, stratum 2). v2 starts 3.01 M at JP against 4.82 M for the plan and 3.70 M at KR against 1.78 M, and the opposite at TW leading (2.12 against 0.84).
  - v2 wastes: 0.60 M disposed at TW leading and 0.78 M at KR, with 0.33 M and 0.37 M stuck at the fabs at the end, and 0.35 M and 0.58 M packaged chips stranded at plants (osat_tw, osat_kr).
  - The plan feeds JP and sells in EU (v2 EU shortage 0.63 against 0.15 for the plan).
  - Wrong fab choice plus stranded stock; foresight is not clearly needed (first big change at week 46). A better allocation of wafer and energy across fabs would help.
- **Episode 30** (gap 189 = chips 92 + shed 88, RSS 0.69, stratum 1). v2 disposes 0.91 M at KR and 0.78 M at JP, and starts 0.35 M lots at JP against 0.13 M for the plan.
  - The shed gap is in EU (66) and KR (20).
  - A chips and shed mix with 5 later events. The waste part (the 1.7 M disposed) is rule-addressable; the EU shed is the same stock-path effect as in item 4.
- **Episode 21** (gap 55 = chips 14 + shed 30, RSS 0.965). Chips are almost equal. The gap is in shed, and in weeks 1-13 it is +57 (the plan sheds less early) and -30 later.
  - osat_my outbound capacity is 0.22 of nominal and constant in all 52 weeks. The plan does not start any lots at JP (v2 1.28 M) and fewer at KR (6.58 M against 10.74 M) and EU mature (0.49 against 2.52).
  - v2 disposes 4.9 M raw units and strands 1.5 M at fabs and 1.8 M packaged at plants.
  - Fab energy v2 13.5 TWh at KR against 8.3 TWh for the plan.
  - A static outlet constraint visible every week; a rule could stop lots into closed outlets. This is the one place where the team's earlier note on braking lots does not apply, and the effect is about 30-55 bn here.
- **Episode 37** (gap 46 = chips -1 + shed 35, RSS 0.95). osat_my outbound is 0.04 constant. The plan starts 3.74 M lots at KR against 9.08 M for v2, 0 at JP against 0.44 M, and 0 at EU.
  - v2 disposes 3.39 M raw at KR and strands 1.18 M packaged at osat_my.
  - Same as episode 21: a rule that sizes lots to a constant outlet could recover most of the 35 bn shed gap.

Episodes 21 and 37 together show shed gap there equals extra fab energy of about 8 TWh (32.6 bn of 35). Across all 40 episodes that link is weak (R² 0.02), so this is an episode-type effect, not a general one.

## 6. Verdict table

Total 113.6 bn per episode (0.01 RSS about 10.7 bn).

| part | bn | rule / reason | basis |
|---|---|---|---|
| (a) recoverable by a rule with week-t information | 25-70 (point about 45) | see the rows below | partly measured, partly inferred |
| (b) needs foresight of unannounced events | 0-20 (point about 3-10) | residual after (a) and (c), bounded by the static test | measured regression; weak (n=40) |
| (c) on paper only or not attributable | 25-70+ | plan model optimism; the replay shows 87.2 | measured identity, split inferred |

Rows making up (a):

| rule | bn | basis |
|---|---|---|
| Ship raw out of fabs at route capacity before storage fills; drain plants; end-of-horizon draining | 10-28 | measured gap in raw shipped (+0.48 M le, +0.37 M mat, 28 bn at full value) and stranded stock (1.37 M le at ends); bounded by closed routes |
| Do not start lots whose outlet capacity is constantly closed or tiny (episodes 21, 37 type) | 0-15 | measured in 2 episodes; across 40, no link of shed to fab energy (R² 0.02), so inferred |
| Keep grid fuel stock high early and run it down at the end (EU, JP, KR, TW) | 0-30 | measured stock path and +7.2 TWh burn; the rule is not identified and the burn path is unreconciled |
| Stop losing the free segment (2.8 TWh) | 3-11 | measured 11.5 bn at v2's level; the team's measured gain on the previous agent was 11 bn for 6.2 TWh, so about 5 bn expected |

(c) detail:
- Measured: the replay of the plan loses 87.2 [78, 96] bn against the plan's own cost in the same scenario. That is mismatch, and it does not come from foresight.
- Measured: the replay beats v2 in 14 of 40 episodes, and v2 is 26.4 [15, 38] bn worse than it on average. Best-of(v2, replay) is 81.8 bn above the plan.
- So an executable foresight policy exists only at 26 bn above v2 (0.027 RSS). Whether a closed-loop foresight controller could recover more than that is not known; I cannot tell.
- The "other" 7.4 bn (tariff, holding, freight, disposal costs) is not attributed; it is in (c) for lack of a split.

**Does a 0.05 RSS (55 bn) gain remain for rules?** Cannot tell. The rows of (a) are not independent and overlap, and I did not test any rule by simulation. A realistic reading is that rules can add 0.02-0.05; 0.05 or more is possible only if the stock-path rule and the waste rules both work.

## Contradictions with the team's notes

1. FINDINGS says gas left at the end is unavoidable and a single-grid search gives at most 0.04 week of burn. The plan ends with 1.6 K lng at grids against v2's 11.9 K (and 8.0 K at terminals). The plan does it by holding more stock mid-game, not by an end rule; the single-grid model may miss that.
2. `report`/FINDINGS "the plan sells 66 bn more chip_le in US and EU and 20 bn less in JP" reads as an allocation gain. Prices are equal across markets, so it is tie-breaking; the real difference is +0.92 M total chip_le sold.
3. FINDINGS and `pull.md`: braking lots loses sales. That holds on average, but in static episodes with a tiny outlet (21, 37) the plan starts a third to half of v2's lots at no sales loss, and it saves 30-50 bn in shed.
4. FINDINGS "the plan delivers the same fuel; the difference is when it burns" is correct, but the burn difference is also 7.2 TWh more burned (stock at end), not only timing.
5. brief.md item 4 puts v2's lost free energy at 6.2 TWh (that was the base agent); v2 itself loses 2.8 TWh.
6. `v2_small_444.txt` lists plan lots against agent lots per fab as "where the plan starts fewer"; in my data the plan produces 1.89 M fewer chip_le at the fabs but ships 0.48 M more.
7. FINDINGS says the gap goes largely to foresight; my static test finds no dependence on change (slope n.s., intercept about 111 of 113.6). This conflicts with "needs foresight" being a large share, within the limits above.

## Checked afterwards by the lead

Three readings of this report were corrected against the same arrays (the numbers above are right, their
interpretation was not):

- Item 4, "the plan delivers the same fuel and burns 7 TWh more of it by ending with less stock". The stock v2 holds at
  the grids at the end (11.9 thousand) is what it moved from the terminals in the last week (10.6 thousand), which can
  no longer be burned; without that transfer it would end with 1.3 thousand against the plan's 1.6. In weeks 1 to 51
  the plan moves 5.9 thousand more gas into the grids (EU +3.8, KR +1.7, JP +0.4, TW 0) and 2.2 thousand more crude
  into JP: it lands more fuel, it does not burn the same fuel better. Contradiction 1 with the team's notes therefore
  does not hold: the gas left at the end is the last two weeks' arrivals, as the notes say.
- "The plan's burn path does not match the ration formula": week by week the plan burns less gas than the simulator
  would at the plan's own stock, by 8 to 20 thousand per grid and episode, in 15 to 26% of the weeks in which that
  grid sheds, and never more. The plan keeps gas at a grid unburned; the simulator cannot. Its timing of the burn is
  not something to copy, and the row "keep grid fuel stock high early and run it down at the end, 0 to 30 bn" of the
  verdict table should read "land more gas at EU and KR and more crude at JP; source not identified".
- Item 2: the score does depend a little on what changes later. On 256 episodes of root 111 the third of the episodes
  with the largest loss of grid output after week 1 scores 0.800 against 0.848 in the lowest third.
