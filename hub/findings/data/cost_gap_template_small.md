# Where `template` loses to the clairvoyant plan: small, 64 episodes of root 111

USD billions per episode, means over the episodes. Written by `team/experiments/cost_gap.py`.

Check against the cached references: the largest difference of an episode's cost is $0.00.

## 1. By cost component

| component | naive | template | clairvoyant | template - clairvoyant | share of the gap % |
| --- | --- | --- | --- | --- | --- |
| freight | 2.45 | 3.78 | 1.21 | 2.57 | 0.46 |
| war_risk | 0.03 | 0.06 | 0.04 | 0.02 | 0.00 |
| tariff | 6.25 | 8.66 | 6.34 | 2.32 | 0.42 |
| holding | 7.70 | 8.34 | 6.22 | 2.12 | 0.38 |
| queue_holding | 1.55 | 5.84 | 0.23 | 5.61 | 1.01 |
| shortage | 1,156.77 | 1,159.21 | 728.28 | 430.93 | 77.21 |
| disposal | 4.63 | 12.57 | 0.30 | 12.27 | 2.20 |
| shed | 2,240.31 | 1,791.82 | 1,686.93 | 104.88 | 18.79 |
| end stock credit | -2.40 | -3.37 | -0.76 | -2.61 | -0.47 |
| **total** | 3,417.30 | 2,986.90 | 2,428.78 | 558.12 | 100.00 |

Score on these episodes, unweighted: 0.4354.

## 2. By harm level

| level | episodes | naive - clairvoyant | template - clairvoyant | score | largest parts of the gap |
| --- | --- | --- | --- | --- | --- |
| 1 | 25 | 953.61 | 687.15 | 0.28 | shortage +565.6, shed +98.8, disposal +14.1 |
| 2 | 25 | 987.41 | 496.68 | 0.50 | shortage +369.1, shed +105.4, disposal +11.7 |
| 3 | 8 | 1,187.42 | 475.26 | 0.60 | shortage +339.3, shed +114.5, disposal +9.5 |
| 4 | 6 | 873.44 | 386.97 | 0.56 | shortage +249.9, shed +115.3, disposal +11.0 |

## 3. By week

The weekly cost without the end stock credit; a week's gap can be negative (the plan pays early).

| weeks | naive | template | clairvoyant | template - clairvoyant | largest parts of the gap |
| --- | --- | --- | --- | --- | --- |
| 1-13 | 519.27 | 524.87 | 431.42 | 93.45 | shortage +58.6, shed +27.3, disposal +4.8 |
| 14-26 | 858.03 | 750.57 | 642.56 | 108.01 | shortage +81.5, shed +20.8, disposal +3.3 |
| 27-39 | 993.49 | 840.16 | 678.34 | 161.82 | shortage +138.2, shed +18.6, disposal +2.0 |
| 40-52 | 1,048.92 | 874.67 | 677.22 | 197.44 | shortage +152.7, shed +38.2, queue_holding +2.4 |

## 4. What the agent asks for and what moves

Of the volume `template` asks for, 18% is executed; the rest is cut
by the week's capacity, a closed strait or missing stock.

### By commodity (dispatch slots only, units of the commodity)

| commodity | asked / week | executed / week | clairvoyant / week | executed / clairvoyant |
| --- | --- | --- | --- | --- |
| lng | 224,387.56 | 25,338.12 | 21,028.50 | 1.20 |
| crude | 179,828.28 | 3,298.03 | 2,404.80 | 1.37 |
| nucfuel | 12,387.53 | 12,020.82 | 942.89 | 12.75 |
| wafer | 3,229,476.08 | 578,278.52 | 193,909.44 | 2.98 |
| chip_le_raw | 1,028,181.83 | 175,234.19 | 254,606.53 | 0.69 |
| chip_mat_raw | 626,514.96 | 71,410.24 | 86,354.73 | 0.83 |
| chip_le | 412,550.70 | 141,067.03 | 296,883.14 | 0.48 |
| chip_mat | 558,076.15 | 94,209.07 | 115,138.76 | 0.82 |

### By transport mode

| mode | slots | template: mean use of the week's capacity % | clairvoyant % |
| --- | --- | --- | --- |
| air | 36 | 28.15 | 23.10 |
| pipeline | 14 | 17.66 | 4.67 |
| sea | 58 | 23.78 | 20.94 |

### Slots where the clairvoyant plan ships less than `template`

| slot | commodity | template: use of capacity % | clairvoyant % |
| --- | --- | --- | --- |
| air.ct.mat_jp_wafer.fab_jp_memory_1 | wafer | 66.34 | 0.00 |
| pipe.tb.src_kz_uranium.grid_eu | nucfuel | 62.92 | 5.11 |
| pipe.tb.src_kz_uranium.grid_kr | nucfuel | 60.19 | 3.85 |
| pipe.tb.src_kz_uranium.grid_jp | nucfuel | 56.49 | 4.90 |
| sea.ct.mat_jp_wafer.fab_jp_memory_1 | wafer | 66.36 | 15.66 |
| air.ct.mat_de_wafer.fab_eu_mature_1 | wafer | 48.20 | 3.70 |
| air.ct.mat_de_wafer.fab_eu_leading_1 | wafer | 48.20 | 7.22 |
| air.ct.mat_ua_neon.fab_eu_mature_1 | wafer | 54.32 | 16.20 |
| air.ct.mat_ua_neon.fab_tw_leading_1 | wafer | 51.01 | 15.23 |
| air.ct.mat_ua_neon.fab_eu_leading_1 | wafer | 48.47 | 14.59 |
| sea.ct.mat_de_wafer.fab_eu_mature_1 | wafer | 48.45 | 17.86 |
| sea.tb.src_us_lng.chk_panama (lane lane.src_us_lng.term_jp) | lng | 30.21 | 0.00 |
| air.ct.mat_jp_wafer.fab_tw_leading_1 | wafer | 55.41 | 25.90 |
| sea.ct.mat_de_wafer.fab_eu_leading_1 | wafer | 48.45 | 20.42 |
| air.ct.mat_jp_wafer.fab_kr_memory_1 | wafer | 53.59 | 25.81 |

### Slots where the clairvoyant plan ships more than `template`

| slot | commodity | template: use of capacity % | clairvoyant % |
| --- | --- | --- | --- |
| air.ct.osat_tw.sink_eu | chip_le | 25.41 | 63.65 |
| air.ct.osat_tw.sink_us | chip_le | 21.92 | 59.32 |
| air.ct.osat_tw.sink_jp | chip_le | 24.63 | 61.00 |
| sea.ct.osat_my.sink_us | chip_mat | 19.78 | 52.97 |
| sea.ct.osat_tw.sink_jp | chip_mat | 22.12 | 53.33 |
| air.ct.osat_kr.sink_us | chip_le | 33.68 | 64.35 |
| sea.tb.src_qa_lng.chk_hormuz (lane lane.src_qa_lng.term_tw) | lng | 14.42 | 44.19 |
| air.ct.osat_kr.sink_eu | chip_le | 35.98 | 64.28 |
| sea.ct.osat_tw.sink_us | chip_mat | 19.26 | 46.91 |
| sea.tb.src_gulf_crude.chk_hormuz (lane lane.src_gulf_crude.term_kr) | crude | 16.62 | 41.93 |
| air.ct.osat_my.sink_jp | chip_le | 26.83 | 51.37 |
| air.ct.osat_my.sink_us | chip_le | 21.55 | 45.85 |
| air.ct.osat_my.sink_eu | chip_le | 22.83 | 46.36 |
| sea.ct.fab_kr_memory_1.osat_kr | chip_le_raw | 21.77 | 44.03 |
| air.ct.osat_kr.sink_jp | chip_le | 37.40 | 59.05 |
