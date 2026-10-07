# The best plan an agent could follow: small, 32 episodes of root 111

The board's clairvoyant plan may power a fab while its grid sheds base load; the simulator serves the base load first. `plan` below is the clairvoyant plan with that rule (a mixed-integer program), `relaxed` the board's. Written by `lab/anastasiia/stats_lab/plan_stats.py`.

Solves: 128 s on average; 12 of 32 stopped by the time limit (largest gap 0.066), their plan is feasible and its cost an upper bound.

## 1. The score such a plan reaches

Check against the cached references: the largest difference of the relaxed plan's cost is $0.00.

|  | score | level 1 | level 2 | level 3 | level 4 | what it is |
| --- | --- | --- | --- | --- | --- | --- |
| plan | 0.93 | 0.90 | 0.97 | 0.94 | 0.90 | the plan with base load first (its own cost) |
| plan, replayed | 0.83 | 0.80 | 0.87 | 0.86 | 0.77 | its orders played blindly in the simulator |
| relaxed, replayed | 0.72 | 0.64 | 0.80 | 0.79 | 0.70 | the board's plan's orders played blindly |

Episodes per harm level: 1: 10, 2: 14, 3: 6, 4: 2.

## 2. Grids: when the fabs get power

A fab runs only on what its grid delivers above the base load. `powered` is the share of weeks in which the plan gives the grid's fabs any power; `shed` the base load not served, GWh per week.

| grid | base load % of output | plan: powered % | plan: shed | plan: powered while shedding % | relaxed: powered % | relaxed: shed | relaxed: powered while shedding % | plan: weeks in a powered stretch (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_tw | 92.97 | 40.02 | 391.98 | 0.00 | 56.37 | 387.00 | 26.32 | 2.00 |
| grid_kr | 96.51 | 36.42 | 777.69 | 0.00 | 53.55 | 822.63 | 31.67 | 1.00 |
| grid_jp | 98.99 | 9.38 | 2,917.75 | 0.00 | 25.12 | 2,913.70 | 23.56 | 1.00 |
| grid_eu | 99.59 | 29.45 | 3,762.66 | 0.00 | 52.94 | 3,775.30 | 27.16 | 1.00 |

## 3. Fabs: lots started, % of nominal capacity

| fab | product | plan | relaxed | plan: weeks with a start % |
| --- | --- | --- | --- | --- |
| fab_tw_leading_1 | chip_le_raw | 34.63 | 35.94 | 49.45 |
| fab_tw_mature_1 | chip_mat_raw | 13.34 | 18.31 | 31.02 |
| fab_kr_memory_1 | chip_le_raw | 36.40 | 40.79 | 46.25 |
| fab_jp_memory_1 | chip_le_raw | 10.39 | 22.20 | 12.11 |
| fab_eu_leading_1 | chip_le_raw | 23.40 | 28.55 | 33.12 |
| fab_eu_mature_1 | chip_mat_raw | 18.29 | 25.30 | 31.72 |

Weeks 1 to 40: a lot started later cannot be sold before the end.

## 4. Where the plan sends each stock

Per origin and commodity: the plan's mean weekly flow to each destination and its share, beside the share 'send the maximum' would give (the slots' nominal capacities). `open %` is the share of weeks the slot's route is not prohibited; `use %` the plan's flow over the first edge's capacity in those weeks. Tanker cargo (lng, crude) sent through a strait is left out: the program pools it at the strait and decides its destination there, so the lane it was dispatched on says nothing; the next table has what reaches each grid.

| origin | commodity | destination | slots | plan: flow / week | plan: share % | send-the-maximum share % | open % | use % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| src_au_lng | lng | term_jp | 1 | 1,801.37 | 73.23 | 71.59 | 45.73 | 91.64 |
| src_au_lng | lng | term_kr | 1 | 658.61 | 26.77 | 28.41 | 45.13 | 90.24 |
| src_ru_gas | lng | term_jp | 1 | 208.74 | 100.00 | 21.33 | 24.46 | 83.76 |
| src_ru_gas | lng | grid_eu | 1 | 0.00 | 0.00 | 78.67 | 0.00 | 0.00 |
| src_kz_uranium | nucfuel | grid_eu | 1 | 137.86 | 66.00 | 71.62 | 57.93 | 1.67 |
| src_kz_uranium | nucfuel | grid_kr | 1 | 39.77 | 19.04 | 19.76 | 68.09 | 1.68 |
| src_kz_uranium | nucfuel | grid_jp | 1 | 31.26 | 14.96 | 8.62 | 64.18 | 2.90 |
| mat_jp_wafer | wafer | fab_kr_memory_1 | 2 | 49,754.11 | 59.06 | 31.56 | 75.21 | 33.28 |
| mat_jp_wafer | wafer | fab_tw_leading_1 | 2 | 14,310.84 | 16.99 | 14.30 | 88.49 | 31.72 |
| mat_jp_wafer | wafer | fab_tw_mature_1 | 2 | 10,169.12 | 12.07 | 24.36 | 79.84 | 33.16 |
| mat_jp_wafer | wafer | fab_jp_memory_1 | 2 | 10,013.71 | 11.89 | 29.78 | 100.00 | 2.90 |
| mat_de_wafer | wafer | fab_eu_mature_1 | 2 | 13,259.42 | 82.85 | 87.99 | 100.00 | 7.09 |
| mat_de_wafer | wafer | fab_eu_leading_1 | 2 | 2,745.59 | 17.15 | 12.01 | 100.00 | 12.23 |
| mat_ua_neon | wafer | fab_kr_memory_1 | 5 | 25,962.35 | 47.62 | 44.52 | 71.92 | 5.82 |
| mat_ua_neon | wafer | fab_tw_leading_1 | 4 | 18,939.13 | 34.74 | 33.01 | 68.81 | 4.99 |
| mat_ua_neon | wafer | fab_eu_mature_1 | 2 | 8,323.71 | 15.27 | 11.56 | 71.54 | 10.81 |
| mat_ua_neon | wafer | fab_eu_leading_1 | 2 | 1,297.91 | 2.38 | 10.91 | 71.72 | 10.93 |
| fab_tw_leading_1 | chip_le_raw | osat_tw | 2 | 50,487.28 | 86.63 | 79.41 | 100.00 | 19.43 |
| fab_tw_leading_1 | chip_le_raw | osat_my | 2 | 7,790.10 | 13.37 | 20.59 | 76.74 | 41.61 |
| fab_kr_memory_1 | chip_le_raw | osat_kr | 2 | 76,524.61 | 57.58 | 43.65 | 100.00 | 19.19 |
| fab_kr_memory_1 | chip_le_raw | osat_my | 3 | 56,385.51 | 42.42 | 56.35 | 70.29 | 27.02 |
| osat_my | chip_le | sink_us | 1 | 79,899.09 | 63.32 | 42.86 | 57.57 | 68.89 |
| osat_my | chip_le | sink_eu | 1 | 29,876.67 | 23.68 | 15.43 | 61.12 | 79.85 |
| osat_my | chip_le | sink_jp | 1 | 16,408.01 | 13.00 | 12.00 | 66.89 | 68.22 |
| osat_my | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_tw | chip_le | sink_us | 1 | 38,439.56 | 67.69 | 42.86 | 77.58 | 78.17 |
| osat_tw | chip_le | sink_eu | 1 | 13,323.18 | 23.46 | 15.43 | 70.07 | 79.21 |
| osat_tw | chip_le | sink_jp | 1 | 5,026.43 | 8.85 | 12.00 | 65.62 | 80.75 |
| osat_tw | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_kr | chip_le | sink_us | 1 | 56,655.12 | 65.10 | 42.86 | 66.71 | 84.25 |
| osat_kr | chip_le | sink_eu | 1 | 19,717.99 | 22.66 | 15.43 | 62.38 | 88.63 |
| osat_kr | chip_le | sink_jp | 1 | 10,660.58 | 12.25 | 12.00 | 76.26 | 78.08 |
| osat_kr | chip_le | sink_cn | 1 | 0.00 | 0.00 | 29.70 | 0.00 | 0.00 |
| osat_my | chip_mat | sink_us | 2 | 42,603.52 | 64.07 | 43.82 | 63.37 | 43.98 |
| osat_my | chip_mat | sink_eu | 2 | 10,705.40 | 16.10 | 15.78 | 80.56 | 22.23 |
| osat_my | chip_mat | sink_cn | 1 | 7,779.54 | 11.70 | 15.53 | 66.47 | 24.32 |
| osat_my | chip_mat | sink_jp | 2 | 5,410.64 | 8.14 | 24.87 | 66.68 | 9.07 |
| osat_tw | chip_mat | sink_us | 2 | 19,025.78 | 56.22 | 47.64 | 77.40 | 30.81 |
| osat_tw | chip_mat | sink_cn | 1 | 7,828.49 | 23.13 | 21.87 | 75.00 | 24.59 |
| osat_tw | chip_mat | sink_eu | 2 | 3,610.54 | 10.67 | 17.15 | 67.58 | 17.73 |
| osat_tw | chip_mat | sink_jp | 2 | 3,377.17 | 9.98 | 13.34 | 70.91 | 37.48 |

Fuel moved into each grid (its terminal-to-grid and pipeline slots), units per week:

| grid | fuel | burn at full output | plan | plan % of burn | relaxed | relaxed % of burn |
| --- | --- | --- | --- | --- | --- | --- |
| grid_tw | lng | 2,176.99 | 1,521.28 | 69.88 | 1,533.30 | 70.43 |
| grid_tw | crude | 108.85 | 55.06 | 50.58 | 54.86 | 50.40 |
| grid_kr | lng | 2,750.00 | 1,699.96 | 61.82 | 1,646.06 | 59.86 |
| grid_kr | crude | 220.00 | 184.53 | 83.88 | 206.47 | 93.85 |
| grid_kr | nucfuel | 3,300.00 | 39.77 | 1.21 | 157.59 | 4.78 |
| grid_jp | lng | 5,400.00 | 2,605.74 | 48.25 | 2,601.19 | 48.17 |
| grid_jp | crude | 900.00 | 451.55 | 50.17 | 476.62 | 52.96 |
| grid_jp | nucfuel | 1,440.00 | 31.26 | 2.17 | 94.09 | 6.53 |
| grid_eu | lng | 8,840.00 | 5,164.74 | 58.42 | 5,163.17 | 58.41 |
| grid_eu | crude | 1,040.00 | 579.36 | 55.71 | 578.99 | 55.67 |
| grid_eu | nucfuel | 11,960.00 | 137.86 | 1.15 | 701.21 | 5.86 |

## 5. Lanes through straits: how much the plan sends against the strait's state

| state of the lane's straits | slot-weeks | plan: use of the first edge % | slot-weeks used at all % |
| --- | --- | --- | --- |
| every strait of the lane fully open | 37259 | 14.01 | 21.46 |
| a strait partly open (0.25 to 1) | 8385 | 12.14 | 20.47 |
| a strait under 0.25 | 4073 | 9.89 | 16.77 |

## 6. Markets: demand served by the plan, units per week

| market | commodity | demand | plan | plan % | relaxed | relaxed % |
| --- | --- | --- | --- | --- | --- | --- |
| sink_us | chip_le | 335,709.33 | 181,362.88 | 54.02 | 199,570.84 | 59.45 |
| sink_us | chip_mat | 94,416.50 | 66,358.82 | 70.28 | 68,529.12 | 72.58 |
| sink_eu | chip_le | 120,855.36 | 65,164.02 | 53.92 | 65,797.29 | 54.44 |
| sink_eu | chip_mat | 33,989.94 | 16,229.67 | 47.75 | 16,141.25 | 47.49 |
| sink_cn | chip_mat | 113,299.80 | 19,061.14 | 16.82 | 29,775.98 | 26.28 |
| sink_jp | chip_le | 93,998.61 | 33,867.03 | 36.03 | 40,655.51 | 43.25 |
| sink_jp | chip_mat | 26,436.62 | 9,568.85 | 36.20 | 9,595.42 | 36.30 |
