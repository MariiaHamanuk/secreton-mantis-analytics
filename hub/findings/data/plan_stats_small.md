# The best plan an agent could follow: small, 40 episodes of root 444

The board's clairvoyant plan may power a fab while its grid sheds base load; the simulator serves the base load first. `plan` below is the clairvoyant plan with that rule (a mixed-integer program), `relaxed` the board's. Written by `team/experiments/plan_stats.py`.

Solves: 51 s on average; 16 of 40 stopped by the time limit (largest gap 0.079), their plan is feasible and its cost an upper bound.

## 1. The score such a plan reaches

Check against the cached references: the largest difference of the relaxed plan's cost is $0.00.

|  | score | level 1 | level 2 | level 3 | level 4 | what it is |
| --- | --- | --- | --- | --- | --- | --- |
| plan | 0.95 | 0.94 | 0.95 | 0.98 | 0.99 | the plan with base load first (its own cost) |
| plan, replayed | 0.87 | 0.86 | 0.88 | 0.87 | 0.91 | its orders played blindly in the simulator |
| relaxed, replayed | 0.79 | 0.79 | 0.82 | 0.75 | 0.82 | the board's plan's orders played blindly |

Episodes per harm level: 1: 18, 2: 12, 3: 5, 4: 5.

## 2. Grids: when the fabs get power

A fab runs only on what its grid delivers above the base load. `powered` is the share of weeks in which the plan gives the grid's fabs any power; `shed` the base load not served, GWh per week.

| grid | base load % of output | plan: powered % | plan: shed | plan: powered while shedding % | relaxed: powered % | relaxed: shed | relaxed: powered while shedding % | plan: weeks in a powered stretch (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_tw | 92.97 | 41.44 | 430.60 | 0.00 | 58.99 | 423.96 | 31.92 | 1.00 |
| grid_kr | 96.51 | 36.63 | 568.72 | 0.00 | 49.28 | 584.58 | 24.23 | 2.00 |
| grid_jp | 98.99 | 15.24 | 2,498.39 | 0.00 | 28.32 | 2,513.74 | 23.41 | 1.00 |
| grid_eu | 99.59 | 12.36 | 4,625.88 | 0.00 | 38.37 | 4,641.88 | 29.13 | 1.00 |

## 3. Fabs: lots started, % of nominal capacity

| fab | product | plan | relaxed | plan: weeks with a start % |
| --- | --- | --- | --- | --- |
| fab_tw_leading_1 | chip_le_raw | 31.01 | 31.01 | 46.50 |
| fab_tw_mature_1 | chip_mat_raw | 24.73 | 28.04 | 39.56 |
| fab_kr_memory_1 | chip_le_raw | 30.71 | 33.46 | 46.94 |
| fab_jp_memory_1 | chip_le_raw | 17.38 | 27.41 | 19.75 |
| fab_eu_leading_1 | chip_le_raw | 6.14 | 15.80 | 9.44 |
| fab_eu_mature_1 | chip_mat_raw | 6.74 | 16.60 | 11.88 |

Weeks 1 to 40: a lot started later cannot be sold before the end.

## 4. Where the plan sends each stock

Per origin and commodity: the plan's mean weekly flow to each destination and its share, beside the share 'send the maximum' would give (the slots' nominal capacities). `open %` is the share of weeks the slot's route is not prohibited; `use %` the plan's flow over the first edge's capacity in those weeks. Tanker cargo (lng, crude) sent through a strait is left out: the program pools it at the strait and decides its destination there, so the lane it was dispatched on says nothing; the next table has what reaches each grid.

| origin | commodity | destination | slots | plan: flow / week | plan: share % | send-the-maximum share % | open % | use % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| src_au_lng | lng | term_jp | 1 | 2,220.94 | 70.98 | 71.59 | 53.70 | 91.08 |
| src_au_lng | lng | term_kr | 1 | 908.02 | 29.02 | 28.41 | 58.32 | 88.54 |
| src_ru_gas | lng | term_jp | 1 | 235.94 | 100.00 | 21.33 | 27.79 | 93.77 |
| src_ru_gas | lng | grid_eu | 1 | 0.00 | 0.00 | 78.67 | 0.00 | 0.00 |
| src_kz_uranium | nucfuel | grid_eu | 1 | 172.47 | 64.16 | 71.62 | 69.76 | 1.83 |
| src_kz_uranium | nucfuel | grid_kr | 1 | 69.18 | 25.74 | 19.76 | 70.67 | 2.50 |
| src_kz_uranium | nucfuel | grid_jp | 1 | 27.17 | 10.11 | 8.62 | 48.08 | 3.30 |
| mat_jp_wafer | wafer | fab_kr_memory_1 | 2 | 41,600.12 | 43.21 | 31.56 | 88.46 | 26.19 |
| mat_jp_wafer | wafer | fab_tw_mature_1 | 2 | 20,663.67 | 21.46 | 24.36 | 81.18 | 25.73 |
| mat_jp_wafer | wafer | fab_jp_memory_1 | 2 | 17,583.34 | 18.26 | 29.78 | 100.00 | 5.12 |
| mat_jp_wafer | wafer | fab_tw_leading_1 | 2 | 16,430.67 | 17.07 | 14.30 | 80.60 | 31.97 |
| mat_de_wafer | wafer | fab_eu_mature_1 | 2 | 4,914.64 | 87.49 | 87.99 | 100.00 | 2.51 |
| mat_de_wafer | wafer | fab_eu_leading_1 | 2 | 702.62 | 12.51 | 12.01 | 100.00 | 2.96 |
| mat_ua_neon | wafer | fab_kr_memory_1 | 5 | 21,059.51 | 58.39 | 44.52 | 50.38 | 8.30 |
| mat_ua_neon | wafer | fab_tw_leading_1 | 4 | 12,954.77 | 35.92 | 33.01 | 49.35 | 9.96 |
| mat_ua_neon | wafer | fab_eu_mature_1 | 2 | 1,890.94 | 5.24 | 11.56 | 56.49 | 4.31 |
| mat_ua_neon | wafer | fab_eu_leading_1 | 2 | 163.21 | 0.45 | 10.91 | 56.68 | 2.15 |
| fab_tw_leading_1 | chip_le_raw | osat_tw | 2 | 47,923.84 | 87.90 | 79.41 | 100.00 | 18.77 |
| fab_tw_leading_1 | chip_le_raw | osat_my | 2 | 6,595.05 | 12.10 | 20.59 | 72.12 | 33.59 |
| fab_kr_memory_1 | chip_le_raw | osat_kr | 2 | 79,988.71 | 66.90 | 43.65 | 100.00 | 19.67 |
| fab_kr_memory_1 | chip_le_raw | osat_my | 3 | 39,569.11 | 33.10 | 56.35 | 69.09 | 18.13 |
| osat_my | chip_le | sink_us | 1 | 79,331.38 | 70.36 | 42.86 | 62.69 | 66.67 |
| osat_my | chip_le | sink_eu | 1 | 16,930.45 | 15.02 | 15.43 | 57.40 | 82.63 |
| osat_my | chip_le | sink_jp | 1 | 16,486.62 | 14.62 | 12.00 | 64.57 | 57.56 |
| osat_my | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_tw | chip_le | sink_us | 1 | 34,806.67 | 64.19 | 42.86 | 68.56 | 78.41 |
| osat_tw | chip_le | sink_eu | 1 | 14,043.64 | 25.90 | 15.43 | 78.75 | 73.99 |
| osat_tw | chip_le | sink_jp | 1 | 5,374.43 | 9.91 | 12.00 | 92.02 | 69.60 |
| osat_tw | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_kr | chip_le | sink_us | 1 | 57,458.78 | 63.29 | 42.86 | 70.87 | 80.37 |
| osat_kr | chip_le | sink_eu | 1 | 20,978.04 | 23.11 | 15.43 | 64.66 | 86.52 |
| osat_kr | chip_le | sink_jp | 1 | 12,346.16 | 13.60 | 12.00 | 71.49 | 70.34 |
| osat_kr | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_my | chip_mat | sink_us | 2 | 30,132.23 | 64.50 | 43.82 | 56.56 | 29.95 |
| osat_my | chip_mat | sink_eu | 2 | 6,929.42 | 14.83 | 15.78 | 78.70 | 16.87 |
| osat_my | chip_mat | sink_cn | 1 | 6,155.77 | 13.18 | 15.53 | 66.06 | 13.49 |
| osat_my | chip_mat | sink_jp | 2 | 3,501.32 | 7.49 | 24.87 | 65.31 | 7.83 |
| osat_tw | chip_mat | sink_us | 2 | 21,078.79 | 48.33 | 47.64 | 67.98 | 36.61 |
| osat_tw | chip_mat | sink_cn | 1 | 13,680.87 | 31.37 | 21.87 | 75.77 | 37.01 |
| osat_tw | chip_mat | sink_eu | 2 | 5,512.42 | 12.64 | 17.15 | 76.61 | 26.39 |
| osat_tw | chip_mat | sink_jp | 2 | 3,338.04 | 7.65 | 13.34 | 84.74 | 39.55 |

Fuel moved into each grid (its terminal-to-grid and pipeline slots), units per week:

| grid | fuel | burn at full output | plan | plan % of burn | relaxed | relaxed % of burn |
| --- | --- | --- | --- | --- | --- | --- |
| grid_tw | lng | 2,176.99 | 1,448.07 | 66.52 | 1,456.53 | 66.91 |
| grid_tw | crude | 108.85 | 49.67 | 45.63 | 50.56 | 46.45 |
| grid_kr | lng | 2,750.00 | 1,972.30 | 71.72 | 1,950.91 | 70.94 |
| grid_kr | crude | 220.00 | 173.97 | 79.08 | 187.58 | 85.26 |
| grid_kr | nucfuel | 3,300.00 | 69.18 | 2.10 | 203.71 | 6.17 |
| grid_jp | lng | 5,400.00 | 3,019.80 | 55.92 | 3,012.43 | 55.79 |
| grid_jp | crude | 900.00 | 497.70 | 55.30 | 503.71 | 55.97 |
| grid_jp | nucfuel | 1,440.00 | 27.17 | 1.89 | 75.52 | 5.24 |
| grid_eu | lng | 8,840.00 | 4,472.66 | 50.60 | 4,472.60 | 50.60 |
| grid_eu | crude | 1,040.00 | 480.83 | 46.23 | 480.83 | 46.23 |
| grid_eu | nucfuel | 11,960.00 | 172.47 | 1.44 | 662.75 | 5.54 |

## 5. Lanes through straits: how much the plan sends against the strait's state

| state of the lane's straits | slot-weeks | plan: use of the first edge % | slot-weeks used at all % |
| --- | --- | --- | --- |
| every strait of the lane fully open | 45328 | 14.64 | 20.71 |
| a strait partly open (0.25 to 1) | 5238 | 11.44 | 19.66 |
| a strait under 0.25 | 6815 | 9.36 | 16.32 |

## 6. Markets: demand served by the plan, units per week

| market | commodity | demand | plan | plan % | relaxed | relaxed % |
| --- | --- | --- | --- | --- | --- | --- |
| sink_us | chip_le | 335,709.33 | 177,916.86 | 53.00 | 187,277.92 | 55.79 |
| sink_us | chip_mat | 94,416.50 | 55,922.07 | 59.23 | 61,094.29 | 64.71 |
| sink_eu | chip_le | 120,855.36 | 54,229.49 | 44.87 | 56,150.67 | 46.46 |
| sink_eu | chip_mat | 33,989.94 | 13,032.23 | 38.34 | 13,301.36 | 39.13 |
| sink_cn | chip_mat | 113,299.80 | 23,306.81 | 20.57 | 31,867.47 | 28.13 |
| sink_jp | chip_le | 93,998.61 | 35,927.33 | 38.22 | 43,932.13 | 46.74 |
| sink_jp | chip_mat | 26,436.62 | 7,639.42 | 28.90 | 10,122.56 | 38.29 |
