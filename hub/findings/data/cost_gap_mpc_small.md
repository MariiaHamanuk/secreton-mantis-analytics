# Where `mpc` loses to the clairvoyant plan: small, 64 episodes of root 111

USD billions per episode, means over the episodes. Written by `team/experiments/cost_gap.py`.

Check against the cached references: the largest difference of an episode's cost is $0.00.

## 1. By cost component

| component | naive | mpc | clairvoyant | mpc - clairvoyant | share of the gap % |
| --- | --- | --- | --- | --- | --- |
| freight | 2.45 | 1.27 | 1.21 | 0.05 | 0.02 |
| war_risk | 0.03 | 0.04 | 0.04 | 0.00 | 0.00 |
| tariff | 6.25 | 4.11 | 6.34 | -2.23 | -0.75 |
| holding | 7.70 | 6.10 | 6.22 | -0.12 | -0.04 |
| queue_holding | 1.55 | 0.09 | 0.23 | -0.13 | -0.05 |
| shortage | 1,156.77 | 1,017.27 | 728.28 | 288.99 | 97.10 |
| disposal | 4.63 | 1.45 | 0.30 | 1.15 | 0.39 |
| shed | 2,240.31 | 1,696.98 | 1,686.93 | 10.05 | 3.38 |
| end stock credit | -2.40 | -0.92 | -0.76 | -0.16 | -0.05 |
| **total** | 3,417.30 | 2,726.39 | 2,428.78 | 297.61 | 100.00 |

Score on these episodes, unweighted: 0.6989.

## 2. By harm level

| level | episodes | naive - clairvoyant | mpc - clairvoyant | score | largest parts of the gap |
| --- | --- | --- | --- | --- | --- |
| 1 | 25 | 953.61 | 396.89 | 0.58 | shortage +398.9, tariff -3.0, disposal +0.8 |
| 2 | 25 | 987.41 | 228.57 | 0.77 | shortage +220.6, shed +9.9, tariff -2.7 |
| 3 | 8 | 1,187.42 | 264.93 | 0.78 | shortage +255.8, shed +8.4, disposal +1.9 |
| 4 | 6 | 873.44 | 215.16 | 0.75 | shortage +160.4, shed +52.3, disposal +1.9 |

## 3. By week

The weekly cost without the end stock credit; a week's gap can be negative (the plan pays early).

| weeks | naive | mpc | clairvoyant | mpc - clairvoyant | largest parts of the gap |
| --- | --- | --- | --- | --- | --- |
| 1-13 | 519.27 | 432.95 | 431.42 | 1.54 | disposal +0.9, shed +0.9, shortage -0.4 |
| 14-26 | 858.03 | 670.43 | 642.56 | 27.87 | shortage +32.8, shed -4.3, tariff -0.6 |
| 27-39 | 993.49 | 794.73 | 678.34 | 116.39 | shortage +120.9, shed -3.4, tariff -1.0 |
| 40-52 | 1,048.92 | 829.19 | 677.22 | 151.96 | shortage +135.8, shed +17.0, tariff -0.8 |

## 4. What the agent asks for and what moves

Of the volume `mpc` asks for, 100% is executed; the rest is cut
by the week's capacity, a closed strait or missing stock.

### By commodity (dispatch slots only, units of the commodity)

| commodity | asked / week | executed / week | clairvoyant / week | executed / clairvoyant |
| --- | --- | --- | --- | --- |
| lng | 21,328.09 | 21,308.90 | 21,028.50 | 1.01 |
| crude | 2,110.92 | 2,110.14 | 2,404.80 | 0.88 |
| nucfuel | 1,642.53 | 1,642.53 | 942.89 | 1.74 |
| wafer | 111,244.74 | 111,174.01 | 193,909.44 | 0.57 |
| chip_le_raw | 154,377.21 | 154,139.77 | 254,606.53 | 0.61 |
| chip_mat_raw | 60,559.48 | 60,519.77 | 86,354.73 | 0.70 |
| chip_le | 192,330.08 | 192,296.16 | 296,883.14 | 0.65 |
| chip_mat | 87,323.59 | 87,275.35 | 115,138.76 | 0.76 |

### By transport mode

| mode | slots | mpc: mean use of the week's capacity % | clairvoyant % |
| --- | --- | --- | --- |
| air | 36 | 15.33 | 23.10 |
| pipeline | 14 | 5.32 | 4.67 |
| sea | 58 | 16.05 | 20.94 |

### Slots where the clairvoyant plan ships less than `mpc`

| slot | commodity | mpc: use of capacity % | clairvoyant % |
| --- | --- | --- | --- |
| pipe.tb.src_kz_uranium.grid_eu | nucfuel | 9.01 | 5.11 |
| air.ct.osat_tw.sink_eu | chip_mat | 12.29 | 9.21 |
| pipe.tb.src_kz_uranium.grid_kr | nucfuel | 6.67 | 3.85 |
| pipe.tb.src_kz_uranium.grid_jp | nucfuel | 7.48 | 4.90 |
| sea.tb.src_us_crude.chk_panama (lane lane.src_us_crude.term_tw) | crude | 51.34 | 49.34 |
| sea.tb.src_qa_lng.chk_hormuz (lane lane.src_qa_lng.term_tw) | lng | 45.35 | 44.19 |
| sea.ct.mat_ua_neon.chk_turkish (lane lane.mat_ua_neon.fab_kr_memory_1) | wafer | 1.75 | 1.03 |
| air.ct.osat_tw.sink_jp | chip_mat | 5.71 | 5.06 |
| air.ct.osat_my.sink_eu | chip_mat | 10.09 | 9.52 |
| sea.tb.src_us_crude.term_eu | crude | 57.90 | 57.37 |
| sea.tb.src_qa_lng.chk_hormuz (lane lane.src_qa_lng.term_kr) | lng | 0.80 | 0.27 |
| sea.ct.mat_ua_neon.chk_turkish (lane lane.mat_ua_neon.fab_tw_leading_1) | wafer | 1.89 | 1.41 |
| sea.tb.src_ru_gas.term_jp | lng | 22.62 | 22.15 |
| tg.tb.term_tw.grid_tw | lng | 25.48 | 25.12 |
| sea.ct.mat_ua_neon.chk_turkish (lane lane.mat_ua_neon.fab_tw_leading_1.cape) | wafer | 0.49 | 0.15 |

### Slots where the clairvoyant plan ships more than `mpc`

| slot | commodity | mpc: use of capacity % | clairvoyant % |
| --- | --- | --- | --- |
| air.ct.osat_kr.sink_us | chip_le | 39.11 | 64.35 |
| air.ct.osat_kr.sink_eu | chip_le | 39.27 | 64.28 |
| air.ct.osat_tw.sink_jp | chip_le | 37.16 | 61.00 |
| sea.ct.osat_tw.sink_jp | chip_mat | 29.55 | 53.33 |
| air.ct.osat_kr.sink_jp | chip_le | 35.39 | 59.05 |
| sea.ct.fab_kr_memory_1.osat_kr | chip_le_raw | 24.65 | 44.03 |
| air.ct.osat_tw.sink_us | chip_le | 41.03 | 59.32 |
| air.ct.osat_tw.sink_eu | chip_le | 46.12 | 63.65 |
| air.ct.osat_my.sink_us | chip_le | 28.43 | 45.85 |
| east.ct.fab_jp_memory_1.osat_my | chip_le_raw | 21.28 | 38.39 |
| sea.ct.mat_jp_wafer.fab_kr_memory_1 | wafer | 11.52 | 28.25 |
| air.ct.osat_my.sink_jp | chip_le | 35.16 | 51.37 |
| sea.ct.osat_my.sink_us | chip_mat | 37.87 | 52.97 |
| sea.ct.fab_jp_memory_1.chk_taiwan (lane lane.fab_jp_memory_1.osat_my) | chip_le_raw | 11.76 | 26.40 |
| air.ct.mat_jp_wafer.fab_kr_memory_1 | wafer | 12.35 | 25.81 |
