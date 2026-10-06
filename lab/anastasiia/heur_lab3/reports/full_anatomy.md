# Full: where the rule agent loses to the clairvoyant plan (6 October)

Root 111: 24 episodes beside the clairvoyant LP, 64 for the agent's own account; scripts and arrays of that day in outputs/heur3/full_anatomy/ (local, not in git)

# v2 on Full: where it loses (root 111; v2 = agents/anastasiia_rules_v2)

Everything below is measured unless marked "inferred". Scripts and logs are in `$MAIN/outputs/heur3/full_anatomy/`.

**Sample sizes.**
- Clairvoyant-LP comparisons use 24 episodes of Full (0-23) and 16 of Small. Run with `diag_save.py`, which wraps `diag.episode` and pickles the raw per-episode data. Output is in `diag_full_24.pkl` and `diag_small_16.pkl`.
- Agent-only accounting uses 64 Full episodes: `account.py --save` writes `v2_full_111_64.npz`, and `account_full_64.txt` is its printout.
- The 64-episode board score is 0.8359 (`costs.py`, levels [32,13,18,1], `costs_full_111.json`). The unweighted per-episode mean is 0.827.
- On the 24 episodes: naive 8,922, v2 6,151 and clairvoyant 5,639 bn. Room (naive minus clairvoyant) is 3,283 bn, so 0.01 RSS is about 33 bn.
- Ranges are given only where a number is not a direct measurement.

## 1. Gap by component (v2 minus clairvoyant, bn per episode)

Run `gap.py full diag_full_24.pkl` and `gap.py small diag_small_16.pkl`. Output is in `gap_full_24.txt` and `gap_small_16.txt`.

| component | Full gap | % of Full room | Small gap | % of Small room |
|---|---|---|---|---|
| shortage | 387.7 | 11.81 | 151.0 | 15.08 |
| shed | 84.1 | 2.56 | 29.6 | 2.95 |
| disposal | 20.0 | 0.61 | 3.2 | 0.31 |
| holding | 16.2 | 0.49 | 0.3 | 0.03 |
| tariff | 2.4 | 0.07 | 1.5 | 0.15 |
| queue holding | 0.9 | 0.03 | 0.1 | 0.01 |
| freight | 0.5 | 0.02 | -0.1 | -0.01 |
| **total** | **512 (15.6 %)** | | **186 (18.5 %)** | |

- Shortage is a smaller share on Full than on Small.
- Disposal is about 2 times and holding about 16 times larger as a share of room on Full. Together that is 36 bn (1.1 % of room) on Full against 3.5 bn (0.3 %) on Small.
- On Full the shed gap is not nil, and it is not "about the clairvoyant's". It is 84 bn (111 bn on the first 8 episodes), 16 % of the whole gap.

**Shortage gap by market (Full, bn, v2 minus clairvoyant):**

| market | gap |
|---|---|
| US le 77.0 + US mat 73.5 | 150.5 |
| CN mat | 41.7 |
| EU le | 38.8 |
| SEA (le 16.5, mat 19.8) | 36.3 |
| JP (le 16.1, mat 18.9) | 35.0 |
| ROW (le 23.6, mat 6.6) | 30.2 |
| KR (le 8.7, mat 20.6) | 29.3 |
| IN (le 15.5, mat 9.2) | 24.7 |
| EU mat | 1.2 |

- By chip type, le is 196 bn and mat is 192 bn.
- US le is served 40 % by v2 and 46 % by the clairvoyant, so its shortage is huge in both and only the 77 bn gap matters.
- Small, for comparison: US le is 99 of the 151 bn.

**Shed gap by grid (Full):**

| grid | gap | note |
|---|---|---|
| TW | +33.0 | positive in 20 of 24 episodes |
| CN | +40.9 | |
| EU | +13.2 | |
| KR | +11.7 | |
| US | +1.7 | |
| IN | +3.5 | |
| JP | -17.8 | |
| SEA | -2.0 | |

- JP and CN swing in opposite directions by up to ±380 bn (episodes 0 and 11). Their net is +23 bn, so most of the CN number is a different split of shared fuel between JP and CN.
- Shed by period (`timing.py`): the gap is +79 bn in weeks 1-26 and about +5 bn net over weeks 27-104. The clairvoyant sheds more late: -26 bn in weeks 79-91.
- Small has the same start-up gap: +58 bn in weeks 1-13, offset by -29 later.
- Shortage gap by period: about 31 bn per 13 weeks for le and 30 for mat, steady from week 27. It is not an early effect.

## 2. Fabs on Full

Run `account.py` (`account_full_64.txt`, `account_small_16.txt`), `perfab.py` and `dispt.py`.

The per-fab table has weeks 1-90 on Full and weeks 1-38 on Small. "No wafers" and "no power" are shares of capacity.

| fab | started | no wafers | no power: shedding grid | lots/cap in no-shed weeks | LP lots/cap (shed-started share) |
|---|---|---|---|---|---|
| tw_leading_1 | .67 | .06 | .22 | .87 | .52 (39 %) |
| tw_mature_1 | .52 | .24 | .20 | .68 | .43 (32 %) |
| tw_mature_2 | .54 | .21 | .20 | .71 | .41 (32 %) |
| kr_leading_1 | .63 | .05 | .28 | .89 | .18 |
| kr_memory_1 | .62 | .07 | .28 | .87 | .57 (40 %) |
| jp_memory_1 | .52 | .02 | .46 | .97 | .50 (62 %) |
| cn_mature_1 | .33 | .01 | .66 | .96 | .62 (74 %) |
| sea_mature_1 | .13 | .03 | .83 | .88 | .28 (96 %) |
| eu_leading_1 | .50 | .20 | .30 | .73 | .31 |
| row_leading_1 | .45 | .25 | .29 | .66 | .48 |
| eu_mature_1 | .45 | .30 | .25 | .65 | .38 |
| us_leading_1 | .73 | .23 | .04 | .76 | .62 |
| us_leading_2 | .66 | .30 | .04 | .69 | .47 |
| us_leading_3 | .52 | .45 | .03 | .54 | .32 |
| us_mature_1 | .35 | .64 | .01 | .36 | .23 |
| us_mature_2 | .68 | .28 | .04 | .71 | .80 |

- The "sliver partly there" column is 0.00-0.05 everywhere. It is never the binding reason.
- Full counterparts of the Small fabs run better than on Small, not worse. Small, "started" for the same families: tw_leading .44, tw_mature .19, kr_memory .31, jp_memory .16, eu_leading .25, eu_mature .22.
- Fabs that are clearly worse on Full than their families:
  - **us_mature_1, us_leading_3, us_leading_2, row, eu_mature**: "no wafers" is .2-.6. v2 deliberately gives them few wafers because their outlets are cut. The clairvoyant starts fewer lots there too, so this is not a loss.
  - **cn_mature_1 and sea_mature_1**: "no power: shedding grid" is .66 and .83. The clairvoyant runs 74 % and 96 % of its lots in weeks when the grid sheds. CN lots are 13.4M for v2 against 24.6M for the clairvoyant. This is a relaxation gain (class c).
- Raw chips disposed at fabs, M units per episode (all numbers below are from `v2_full_111_64.npz`):
  - kr_memory 2.14, jp_memory 1.10, tw_mature_2 1.43, eu_mature 1.27, us_mature_1 0.77.
  - us_leading ×3 total 1.39M, eu_leading 0.50, row_leading 0.28, tw_leading 0.36.
  - Total value if sold is about 335 bn.
- **Timing of that disposal:** weeks 1-12 hold 35 bn, weeks 13-40 hold 86 bn, weeks 41-104 hold 215 bn. On Small it is mostly weeks 1-12 (initial work in process). On Full it is mostly steady-state over-production late in the episode.
- Disposal cost is 18.9 bn per episode (64 episodes). The clairvoyant disposes 0.3 bn.
- Wafers disposed are small, apart from sea_mature 0.50M and kr_memory 0.54M.

## 3. Grids on Full

Run `grids.py full v2_full_111_64.npz`, plus the table in `account_full_64.txt`. Shortfall means cap minus segment, in shed weeks.

| grid | weeks without shed % | shed GWh/ep (bn) | fuel short in shed weeks (GWh/ep) | free segment lost (GWh/wk) |
|---|---|---|---|---|
| tw | 77 | 33k (138) | lng 41k, crude 1.8k | 11 |
| kr | 72 | 64k (263) | lng 66k, crude 3.6k, nuc 1k | 6 |
| jp | 53 | 152k (627) | lng 115k, crude 28k, nuc 4.4k | 1 |
| cn | 34 | 204k (843) | lng 98k, crude 74k, nuc 35k | 3.5 |
| us | 96 | 77k (318) | nuc 40k only | 43 |
| eu | 69 | 98k (403) | lng 25k, crude 28k, nuc 25k | 32 |
| sea | 15 | 304k (1255) | lng 250k, crude 62k | 1.3 |
| in | 92 | 24k (98) | nuc 4.2k | 0 |

- SEA alone carries 32 % of all Full shed.
- Fuel left at the end of the episode is small: gas at grid_jp 5.0k, grid_cn 5.4k, grid_eu 12.2k, grid_us 49.5k (harmless pipe gas). Fuel waiting at straits at the end is 4,704 units. This is small next to a base-load week of any one grid.
- **grid_us** loses almost only nuclear fuel (`nuc2.py`).
  - In 4 of 64 episodes the stock is empty for 43-44 weeks.
  - The clairvoyant's US shed is the same (321.6 against 323.3 bn). These are closed lanes (class b).
  - Gas disposed at grid_us is 595k units per episode and free energy lost is 43 GWh per week. Neither is a loss, because the fuel is not scarce.
- **grid_eu** nuclear: 4 episodes with 5-44 weeks empty.
- **Nuclear in general** (class b): v2's grid nuclear stock never exceeds about 0.5 to 0.57 of storage in any grid.
  - v2 sends at the lane maximum from week 1 (6.8k per week to CN, against a burn of 9k).
  - The clairvoyant delivers 17k GWh more nuclear to CN, 15k to EU, 11k to US and about 2.6k each to JP, KR and IN, but sends it late (weeks 53-91).
  - How much of that is a lane-timing rule is open, class d. The delivered totals suggest it is not larger than the CN, EU and US shed gaps of 41, 13 and 2 bn.
- **Gas and crude delivered to grids** is almost the same for v2 and the clairvoyant (`fuelflow.py`, `fuelflow_full_24.txt`): lng to grid_cn 437.8k against 442.2k, grid_tw 165.6k against 165.6k.
  - The shed gap is therefore a matter of when fuel burns, not of how much arrives.
  - Concentration into on/off weeks (the "hold") is a deliberate trade for chips. It is the same mechanism as the TW start-up shed in weeks 5-9: `earlytw.py` shows grid stock of 0.9-1.4k, below the 2,053 threshold, and ration of about 0.4, while the terminal holds 2.3-4.7k.
- **Rules that do not fit a grid:**
  - **grid_in** has no fabs. v2 sends 64.6k crude to term_in against 37k of burn (the clairvoyant sends 0.8k), and 30.6k is disposed there. Gulf crude is shared with JP, CN and KR, whose crude is short (68 %, 60 %, 78 % burned). Inferred small effect, a few bn at most.
  - **grid_sea**: crude is burned at 19 % and lng at 68 %. The clairvoyant has the same shed, so it is class b.
  - Terminals dispose lng: EU 98k, SEA 62k, CN 54k, JP 21k, TW 15k, KR 9k units per episode. Delivery to grids is equal to the clairvoyant's, so this is over-ordering, not lost deliveries (inferred).

## 4. Chips made and never sold

Run `dispt.py`, `perfab.py`, `stages.py`, `edgeutil.py`, `shared.py`, `idle3.py` and `balance.py` (`balance_full_8.txt`).

Disposal at fabs, with sale value (le at 50.4k, mat at 10.3k per unit):

| stock slot | units/ep | value (bn) |
|---|---|---|
| kr_memory le_raw | 2.14M | 108 |
| jp_memory le_raw | 1.10M | 55 |
| us_leading ×3 | 1.39M | 70 |
| eu_leading | 0.50M | 26 |
| tw_leading | 0.36M | 18 |
| osat_kr le (plant) | 0.32M | 16 |
| tw_mature_2 mat_raw | 1.43M | 15 |
| row_leading | 0.28M | 14 |
| eu_mature mat_raw | 1.27M | 13 |
| us_mature_1 mat_raw | 0.77M | 8 |

The chain over 24 episodes (`stages.py`), v2 against the clairvoyant, in M units per episode:

| stage | le v2 | le LP | mat v2 | mat LP |
|---|---|---|---|---|
| lots | 38.2 | 32.6 | 44.0 | 51.9 |
| raw shipped fab to plant | 36.4 | 38.3 | 45.6 | 59.0 |
| packaged shipped plant to market | 37.6 | 41.3 | 47.9 | 64.0 |
| served | 38.0 | 41.9 | 47.8 | 66.3 |

- The clairvoyant starts fewer le lots and sells more.
- v2's markets hold no stock, and served is about equal to shipped. So sales are limited by what the plants ship out, not by market stock.

**Outlet check, le (`edgeutil.py`, `shared.py`).** Utilisation of the open direct plant-to-market first edges, shared by le and mat:

| plant | v2 | clairvoyant |
|---|---|---|
| osat_my | 89 % | 98 % |
| osat_ph | 84 % | 93 % |
| osat_sg | 78 % | 91 % |
| osat_tw | 93 % | 95 % |

- The clairvoyant ships 1.76M more le and 0.18M more mat through these edges. That is worth +90.6 bn at those four plants.
- By destination, the extra US le is 1.5M units (about 76 bn). This matches the US le shortage gap.
- v2's packaged-chip backlog at plants is about 550k at osat_my and osat_kr, 300k at osat_tw, and the stock is not empty in most weeks.
- Upper bound of what was idle (`idle3.py`): chips in plant stock at the start of a week that could have used free open direct-edge capacity. It is 1.84M le (92 bn) and 4.76M mat at osat_cn (50 bn). This ignores which market needs the chips and chokepoint limits on lane routes. It is a ceiling, not a recoverable number.
- **Fab to plant edges are not the binding limit.** Open first-edge capacity is 22-56 % used by v2 (`outletcap.py`).
  - JP fab: v2 asks 57k per week while starting 75k lots per week, and the raw stock stays at 200-500k until it overflows (`jp.py`).
  - So for JP, KR memory and the five small fabs the outlet existed in the clairvoyant's play, and v2's own raw-shipping rule leaves chips at the fab.
- **Mature chips:** the whole mat gap is mostly CN and SEA lots (class c), not outlets. Excluding CN and SEA, LP and v2 raw mat shipments are equal (27.2M against 27.1M).
- TW plants are saturated by both agents.

## 5. Per-episode view (64 episodes, `per_episode.py`)

- Score distribution: mean 0.827, median 0.841, min 0.548, max 0.960.
- Quantiles 10/25/50/75/90: .731, .780, .841, .880, .913.
- Mean score by harm level: level 1 .846 (n=32), level 2 .826 (13), level 3 .791 (18), level 4 one episode (.846).
- Share of the total gap carried by the worst 8, 16 and 32 episodes by gap: 24 %, 42 %, 69 %. So it is moderately concentrated, but not in a few episodes.

The 8 worst episodes by score:

| episode | level | score | what is common |
|---|---|---|---|
| 0 | 2 | .548 | SEA shed 3,332 bn, KR shed 840, US le shortage 943 |
| 22 | 3 | .570 | SEA 3,043, US shed 2,117 (nuclear), KR 998, US le shortage 961 |
| 45 | 3 | .680 | JP shed 1,410, KR 926; kr_leading/kr_memory lots .11, sea_mature .06 |
| 48 | 1 | .712 | EU shed 976, IN 700 |
| 50 | 1 | .721 | JP shed 2,619; jp_memory lots .02 |
| 35 | 2 | .722 | EU shed 2,957 (nuclear) |
| 39 | 3 | .728 | JP shed 1,287, CN 1,006 |
| 18 | 3 | .736 | SEA shed 3,331, CN 1,588 |

- Common to all of them: a grid with a very large shed (SEA or a nuclear cut at US or EU), US le shortage 550-960 bn, and idle KR, JP, EU or SEA fabs.
- The only strong per-grid correlation of episode score with shed is KR, at -0.60. KR is also where the clairvoyant runs its fab while shedding.

## 6. Ranked losses on Full (24 episodes, v2 minus clairvoyant)

Classes: (a) a rule could recover it with week-t information, (b) routes do not allow it, (c) only the relaxed clairvoyant gets it, (d) unknown. RSS is bn divided by 34.

| item | bn | RSS | class |
|---|---|---|---|
| CN mature lots in shed weeks (raw +11.4M mat) | 119 | 0.035 | c |
| le outlet use (net le plant output 3.7M below the clairvoyant) | 187 | 0.055 | a (upper bound 92) / d |
| of which JP memory 81, KR memory 52, five small fabs 77, TW and KR leading -117 | | | |
| mat remainder (stocks, TW plant overflow, timing) | about 52 | 0.015 | d |
| SEA mature lots | 20 | 0.006 | c |
| shed TW | 33 | 0.010 | d / c (fab-shed trade-off) |
| shed JP+CN net (allocation) | 23 | 0.007 | d |
| shed EU | 13 | 0.004 | d |
| shed KR | 12 | 0.004 | c / d |
| disposal cost | 20 | 0.006 | a |
| holding | 16 | 0.005 | a / b |
| tariff, queue holding, freight | 3.8 | 0.001 | d |

The le outlet row overlaps with the disposal row.

Sum of (a) with an honest range:

| item | range (bn) |
|---|---|
| le outlet gap | 30-60 |
| disposal cost | 8-14 |
| shed (start-up, TW) | 0-25 |
| holding | 0-6 |
| **total** | **38-105 bn, about 0.011-0.031 RSS** |

- Central estimate is about 65 bn, about 0.02 RSS.
- Even the whole non-(c) pool, 512 minus about 139 bn of class c, is 373 bn (about 0.11 RSS). It includes class b and d items, so it is not recoverable.
- Reaching 0.05 (about 170 bn) would need all of the optimistic (a) range plus a share of class d. **It looks out of reach with rules on the identified Full items.**
- Basis: this is an inference from the gap items, not an experiment. The le outlet estimate is the least certain. Its ceiling comes from `idle3.py`.

## Contradictions with the team's notes

1. "Shed is about the clairvoyant's, the whole gap is chips." Not quite. On 24 episodes the shed gap is 84 bn (0.026 RSS), and 111 bn on 8. TW is the systematic part, positive in 20 of 24 episodes, and the gap sits in weeks 1-26. Disposal and holding add 36 bn.
2. "Everything disposed on Full is about 30 bn, at most 0.009 RSS." The disposal cost is 20 bn and holding adds 16 bn. The chips made and disposed have a sale value of about 335 bn, 64 % of it in weeks 41-104. That is steady-state over-production, not initial work in process as on Small.
3. "Five small fabs dispose of half their output because routes are cut to 25 %." The 45 % disposal share matches. But the first fab-to-plant edges are only 11-56 % used by v2 and 22-67 % by the clairvoyant, so v2's raw-shipping rule binds, not the edge. The clairvoyant ships 1.5M more le from these five fabs with fewer lots.
4. "grid_us has fuel always." Gas, yes (pipe). Nuclear is lost for 44 weeks in 4 of 64 episodes. The clairvoyant has the same loss (class b).
5. "grid_sea's crude route is cut to 1-8 %." Confirmed (crude burned 19 %). SEA is also 32 % of all shed, and the clairvoyant's shed there is equal to v2's.
6. "Chips are 300-560 bn per episode." On 24 episodes the shortage gap is 388 bn and the 8-episode value is 399 bn. Both are in the range. About 139 bn of it is CN and SEA lots that need shed-week power (class c).

## Method notes

- My first `costs.py` job was killed, probably by another analyst's `kill` of `costs.py` processes. I restarted it with 2 workers and no overlap, and ran one command at a time afterwards.
- No other lab files, `agents/`, `hub/` or git state were touched. Roots 0, 222 and 333 were not used.
- For overall reproducibility run `$MAIN/.venv/bin/python <script>` in `$MAIN/outputs/heur3/full_anatomy/`.

## Checked afterwards by the lead

The item 'le outlet use: a rule could recover 30-60 bn' was probed the same evening with six switches of the shipping rules (pack_margin 3, pack_cover 12, the cover pass over strait lanes too, raw_extra 4, strait_pref off, raw_waits doubled) on Small root 111 x64 and Full root 111 x32: none helps (0.0000 on Full for five of them, pack_margin 3 -0.0016; on Small raw_extra 4 -0.0096). The plant-to-market pass already ships what the route capacities, as the agent computes them, allow; the difference to the clairvoyant is where and when the chips are made, or the agent's estimate of lane capacity behind a strait, not the shipping switches.
