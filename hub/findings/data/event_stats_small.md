# Disruption statistics: small, 2000 episodes of root 333, 52 weeks each

Drawn from the public generator and read on the trusted side; the signals are the `standard` regime's.
Weeks are 1-based as `observation['week']`. Written by `team/experiments/event_stats.py`.

## 1. Events by type

`carried in` started before week 1 and still acts in the episode; `new` starts inside it. Durations of new
events are the generator's, in weeks, not cut at the episode's end.

| type | carried in / episode | new / episode | episodes with a new one % | weeks p25 | p50 | p75 | p90 | median severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tariff | 5.75 | 3.38 | 83.85 | 26.03 | 51.54 | 101.80 | 189.34 | 1.00 |
| sanction | 19.99 | 3.59 | 88.20 | 33.83 | 126.58 | 361.57 | 767.58 | 1.00 |
| material_outage | 1.14 | 0.21 | 17.10 | 43.62 | 142.99 | 349.53 | 725.17 | 1.00 |
| militarised_closure | 0.73 | 0.46 | 25.15 | 9.04 | 32.47 | 82.79 | 200.69 | 0.65 |
| regional_conflict | 0.94 | 0.27 | 15.70 | 15.74 | 36.19 | 91.84 | 259.97 | 0.60 |
| piracy | 0.10 | 0.28 | 18.00 | 6.52 | 13.84 | 25.26 | 49.50 | 0.10 |
| energy_shock | 0.09 | 1.35 | 48.60 | 1.06 | 2.06 | 4.02 | 7.51 | 0.30 |
| weather_closure | 0.03 | 1.42 | 74.70 | 0.49 | 0.85 | 1.48 | 2.38 | 1.00 |
| port_strike | 0.09 | 1.09 | 66.50 | 1.24 | 2.20 | 4.54 | 10.33 | 0.93 |

### Where new events land

| type | distinct targets | most frequent (share of the type's new events) |
| --- | --- | --- |
| tariff | 8 | KR 14%, TW 14%, SEA 13%, EU 13%, ROW 13%, JP 12% |
| sanction | 51 | bypass.tb.src_gulf_crude.chk_malacca 7%, sea.tb.src_ru_gas.term_jp 7%, sea.tb.src_au_lng.term_jp 3%, sea.tb.src_au_lng.term_kr 3%, pipe.tb.src_kz_uranium.grid_eu 3%, pipe.tb.src_kz_uranium.grid_jp 3% |
| material_outage | 3 | mat_jp_wafer 39%, mat_de_wafer 35%, mat_ua_neon 26% |
| militarised_closure | 4 | chk_malacca 38%, chk_taiwan 34%, chk_hormuz 14%, chk_suez 14% |
| regional_conflict | 13 | RU 15%, UA 12%, ROW 12%, US 11%, TW 11%, AU 10% |
| piracy | 4 | chk_malacca 43%, chk_taiwan 30%, chk_hormuz 14%, chk_suez 13% |
| energy_shock | 4 | grid_tw 27%, grid_kr 25%, grid_jp 24%, grid_eu 24% |
| weather_closure | 7 | chk_taiwan 15%, chk_cape 15%, chk_malacca 15%, chk_suez 14%, chk_hormuz 14%, chk_turkish 14% |
| port_strike | 11 | KR 10%, SEA 10%, TW 10%, AU 9%, RU 9%, US 9% |

### When new events start

| type | weeks 1-13 % | weeks 14-26 % | weeks 27-39 % | weeks 40-52 % |
| --- | --- | --- | --- | --- |
| tariff | 25.17 | 26.01 | 24.09 | 24.73 |
| sanction | 25.39 | 25.37 | 24.61 | 24.63 |
| material_outage | 23.56 | 27.40 | 26.68 | 22.36 |
| militarised_closure | 28.56 | 25.71 | 22.32 | 23.41 |
| regional_conflict | 25.65 | 27.88 | 26.02 | 20.45 |
| piracy | 24.73 | 26.71 | 24.73 | 23.83 |
| energy_shock | 26.61 | 25.83 | 23.76 | 23.80 |
| weather_closure | 24.40 | 24.86 | 25.18 | 25.56 |
| port_strike | 24.83 | 24.51 | 24.92 | 25.74 |

## 2. What the network looks like week by week

`hit` means the week differs from the calm plan. `runs to the end` is the share of spells still running in
the last week. The level is the open fraction, the capacity ratio or the restoration factor (1 is normal).

### Straits (`graph_now.open` < 1)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chk_malacca | 44.15 | 24.12 | 24.80 | 0.47 | 2.00 | 23.00 | 52.00 | 49.79 | 0.31 |
| chk_taiwan | 42.05 | 21.01 | 20.85 | 0.46 | 2.00 | 14.00 | 52.00 | 47.04 | 0.25 |
| chk_suez | 28.95 | 8.52 | 8.50 | 0.31 | 2.00 | 3.00 | 22.00 | 27.20 | 0.37 |
| chk_hormuz | 27.95 | 9.62 | 9.10 | 0.31 | 2.00 | 3.00 | 32.50 | 31.18 | 0.31 |
| chk_cape | 18.90 | 0.86 | 0.60 | 0.21 | 1.00 | 2.00 | 3.00 | 3.83 | 0.50 |
| chk_turkish | 18.30 | 0.82 | 0.65 | 0.20 | 2.00 | 2.00 | 3.00 | 2.51 | 0.50 |
| chk_panama | 17.85 | 0.82 | 0.60 | 0.19 | 2.00 | 2.00 | 3.00 | 2.83 | 0.48 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.85 | 4309 |
| 2 | 0.66 | 3596 |
| 3 | 0.81 | 2335 |
| 4 | 0.91 | 1881 |
| 6 | 0.98 | 1636 |
| 8 | 0.99 | 1546 |
| 12 | 0.99 | 1417 |
| 20 | 0.99 | 1267 |


Of the strait-weeks below 1, 10% are fully closed (open fraction 0).

Straits disrupted in the same week (share of weeks): 0: 50.7%, 1: 35.4%, 2: 11.7%, 3: 1.9%.

War-risk surcharge in force (share of strait-weeks): class 1: 5.6%, class 2: 3.3%.

### New prohibitions per edge (`graph_now.prohibited`)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bypass.tb.src_gulf_crude.chk_malacca | 79.70 | 74.48 | 74.60 | 0.80 | 52.00 | 52.00 | 52.00 | 92.46 | 1.00 |
| sea.tb.src_ru_gas.term_jp | 75.00 | 69.97 | 69.85 | 0.75 | 52.00 | 52.00 | 52.00 | 93.30 | 1.00 |
| sea.tb.src_au_lng.term_jp | 51.50 | 46.36 | 46.60 | 0.52 | 52.00 | 52.00 | 52.00 | 89.43 | 1.00 |
| sea.tb.src_au_lng.term_kr | 51.25 | 46.83 | 46.60 | 0.51 | 52.00 | 52.00 | 52.00 | 90.96 | 1.00 |
| pipe.tb.src_kz_uranium.grid_kr | 43.00 | 38.69 | 38.25 | 0.43 | 52.00 | 52.00 | 52.00 | 89.44 | 1.00 |
| pipe.tb.src_kz_uranium.grid_eu | 43.00 | 37.98 | 38.15 | 0.43 | 52.00 | 52.00 | 52.00 | 87.62 | 1.00 |
| sea.tb.src_us_crude.term_eu | 42.15 | 37.95 | 37.65 | 0.42 | 52.00 | 52.00 | 52.00 | 90.41 | 1.00 |
| sea.ct.osat_my.chk_taiwan | 42.00 | 37.30 | 36.45 | 0.42 | 52.00 | 52.00 | 52.00 | 88.73 | 1.00 |
| pipe.tb.src_kz_uranium.grid_jp | 41.10 | 35.77 | 35.50 | 0.41 | 52.00 | 52.00 | 52.00 | 87.50 | 1.00 |
| sea.tb.src_us_lng.chk_panama | 40.95 | 36.74 | 36.45 | 0.41 | 52.00 | 52.00 | 52.00 | 89.67 | 1.00 |
| sea.ct.osat_my.sink_us | 40.80 | 36.30 | 36.25 | 0.41 | 52.00 | 52.00 | 52.00 | 88.47 | 1.00 |
| sea.tb.src_us_crude.chk_panama | 40.25 | 35.29 | 35.35 | 0.40 | 52.00 | 52.00 | 52.00 | 87.36 | 1.00 |
| sea.tb.src_us_lng.term_eu | 40.00 | 36.29 | 36.00 | 0.40 | 52.00 | 52.00 | 52.00 | 89.29 | 1.00 |
| air.ct.osat_my.sink_jp | 39.15 | 34.78 | 34.85 | 0.39 | 52.00 | 52.00 | 52.00 | 88.18 | 1.00 |
| air.ct.osat_my.sink_us | 38.35 | 34.44 | 34.55 | 0.38 | 52.00 | 52.00 | 52.00 | 89.45 | 1.00 |
| air.ct.osat_my.sink_eu | 38.35 | 33.70 | 33.35 | 0.38 | 52.00 | 52.00 | 52.00 | 88.28 | 1.00 |
| air.ct.osat_kr.sink_us | 36.45 | 32.31 | 32.30 | 0.36 | 52.00 | 52.00 | 52.00 | 87.26 | 1.00 |
| sea.ct.fab_kr_memory_1.chk_taiwan | 35.50 | 31.82 | 31.50 | 0.36 | 52.00 | 52.00 | 52.00 | 89.17 | 1.00 |
| air.ct.osat_kr.sink_eu | 35.10 | 31.61 | 31.65 | 0.35 | 52.00 | 52.00 | 52.00 | 90.06 | 1.00 |
| air.ct.fab_kr_memory_1.osat_my | 35.05 | 31.30 | 31.40 | 0.35 | 52.00 | 52.00 | 52.00 | 88.16 | 1.00 |
| east.ct.fab_kr_memory_1.osat_my | 34.90 | 30.91 | 30.50 | 0.35 | 52.00 | 52.00 | 52.00 | 87.29 | 1.00 |
| air.ct.fab_eu_mature_1.osat_my | 34.75 | 29.67 | 29.90 | 0.35 | 48.00 | 52.00 | 52.00 | 85.49 | 1.00 |
| sea.ct.fab_eu_leading_1.chk_suez | 34.65 | 30.46 | 31.15 | 0.35 | 52.00 | 52.00 | 52.00 | 86.74 | 1.00 |
| cape.ct.fab_eu_mature_1.chk_cape | 34.60 | 30.35 | 30.50 | 0.35 | 52.00 | 52.00 | 52.00 | 87.36 | 1.00 |
| air.ct.osat_kr.sink_jp | 33.95 | 29.64 | 29.60 | 0.34 | 50.75 | 52.00 | 52.00 | 85.09 | 1.00 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 1.00 | 35128 |
| 2 | 1.00 | 34899 |
| 3 | 1.00 | 34701 |
| 4 | 1.00 | 34490 |
| 6 | 1.00 | 34122 |
| 8 | 1.00 | 33762 |
| 12 | 1.00 | 33110 |
| 20 | 1.00 | 31859 |


### Capacity cuts per edge (`graph_now.u` < nominal)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sea.tb.chk_taiwan.term_jp | 85.65 | 79.78 | 79.95 | 0.87 | 52.00 | 52.00 | 52.00 | 91.74 | 0.06 |
| sea.tb.chk_suez.term_eu | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| turnback.tb.chk_suez.term_eu | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| sea.ct.chk_suez.sink_eu | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| cape.tb.chk_cape.term_eu | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| sea.ct.chk_turkish.fab_eu_leading_1 | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| sea.ct.chk_turkish.fab_eu_mature_1 | 82.45 | 75.52 | 75.50 | 0.83 | 52.00 | 52.00 | 52.00 | 90.75 | 0.06 |
| sea.ct.fab_jp_memory_1.chk_taiwan | 82.10 | 75.76 | 75.75 | 0.83 | 52.00 | 52.00 | 52.00 | 91.01 | 0.06 |
| sea.tb.chk_malacca.term_tw | 80.55 | 72.79 | 73.05 | 0.81 | 52.00 | 52.00 | 52.00 | 88.55 | 0.25 |
| sea.ct.chk_malacca.fab_tw_leading_1 | 80.55 | 72.79 | 73.05 | 0.81 | 52.00 | 52.00 | 52.00 | 88.55 | 0.25 |
| sea.ct.osat_tw.sink_jp | 78.95 | 69.34 | 69.25 | 0.82 | 52.00 | 52.00 | 52.00 | 85.51 | 0.25 |
| sea.ct.chk_malacca.chk_taiwan | 78.10 | 72.54 | 72.80 | 0.79 | 52.00 | 52.00 | 52.00 | 91.46 | 0.25 |
| sea.tb.chk_malacca.chk_taiwan | 78.10 | 72.54 | 72.80 | 0.79 | 52.00 | 52.00 | 52.00 | 91.46 | 0.25 |
| sea.ct.chk_taiwan.sink_jp | 76.60 | 69.02 | 69.05 | 0.78 | 52.00 | 52.00 | 52.00 | 88.64 | 0.25 |
| air.ct.osat_tw.sink_jp | 74.25 | 68.73 | 68.80 | 0.75 | 52.00 | 52.00 | 52.00 | 91.82 | 0.25 |
| sea.ct.fab_tw_leading_1.osat_my | 69.00 | 55.59 | 55.15 | 0.71 | 33.00 | 52.00 | 52.00 | 77.92 | 0.25 |
| east.tb.chk_malacca.term_jp | 67.80 | 59.33 | 59.45 | 0.69 | 52.00 | 52.00 | 52.00 | 86.27 | 0.25 |
| sea.ct.mat_jp_wafer.fab_tw_leading_1 | 67.10 | 54.26 | 54.65 | 0.70 | 29.00 | 52.00 | 52.00 | 78.14 | 0.25 |
| sea.ct.mat_jp_wafer.fab_tw_mature_1 | 67.10 | 54.26 | 54.65 | 0.70 | 29.00 | 52.00 | 52.00 | 78.14 | 0.25 |
| sea.ct.chk_taiwan.osat_my | 64.75 | 55.08 | 54.55 | 0.66 | 49.00 | 52.00 | 52.00 | 83.28 | 0.25 |
| sea.ct.fab_eu_mature_1.chk_suez | 63.40 | 53.76 | 54.65 | 0.64 | 52.00 | 52.00 | 52.00 | 83.54 | 0.25 |
| cape.ct.fab_eu_mature_1.chk_cape | 63.40 | 53.76 | 54.65 | 0.64 | 52.00 | 52.00 | 52.00 | 83.54 | 0.25 |
| east.tb.chk_malacca.term_kr | 62.85 | 53.60 | 53.55 | 0.64 | 51.00 | 52.00 | 52.00 | 83.35 | 0.25 |
| east.ct.chk_malacca.fab_kr_memory_1 | 62.85 | 53.60 | 53.55 | 0.64 | 51.00 | 52.00 | 52.00 | 83.35 | 0.25 |
| sea.ct.fab_eu_leading_1.chk_suez | 62.85 | 53.28 | 52.95 | 0.64 | 52.00 | 52.00 | 52.00 | 84.52 | 0.25 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.99 | 74899 |
| 2 | 0.96 | 73697 |
| 3 | 0.97 | 70556 |
| 4 | 0.98 | 68051 |
| 6 | 0.99 | 65675 |
| 8 | 0.99 | 64240 |
| 12 | 1.00 | 62346 |
| 20 | 1.00 | 59345 |


Capacity is below nominal in 25% of edge-weeks; of those, by level: 0.25: 62%, 0.06: 24%, below 0.06: 14%. A sanction on one commodity cuts the other edges between the same two regions to 0.25 of their
capacity, and the cuts multiply (0.25, 0.06, 0.016, ...). A closed strait cuts its lanes too.

### Freight surcharges per edge (`graph_now.c` > nominal)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sea.ct.chk_malacca.chk_taiwan | 18.85 | 6.55 | 6.30 | 0.20 | 6.00 | 12.00 | 24.00 | 38.17 | 1.10 |
| sea.tb.chk_malacca.chk_taiwan | 18.85 | 6.55 | 6.30 | 0.20 | 6.00 | 12.00 | 24.00 | 38.17 | 1.10 |
| sea.ct.chk_malacca.chk_suez | 15.55 | 5.50 | 5.30 | 0.16 | 6.00 | 13.00 | 26.00 | 36.45 | 1.10 |
| sea.ct.chk_suez.chk_malacca | 15.55 | 5.50 | 5.30 | 0.16 | 6.00 | 13.00 | 26.00 | 36.45 | 1.10 |
| sea.tb.chk_hormuz.chk_malacca | 15.50 | 5.60 | 5.50 | 0.16 | 6.00 | 13.00 | 26.00 | 37.69 | 1.10 |
| lombok.ct.chk_suez.chk_taiwan | 12.75 | 4.02 | 3.80 | 0.13 | 5.00 | 12.00 | 22.00 | 31.42 | 1.10 |
| lombok.tb.chk_hormuz.chk_taiwan | 12.60 | 4.13 | 3.75 | 0.13 | 5.00 | 12.00 | 22.00 | 33.20 | 1.10 |
| bypass.tb.src_gulf_crude.chk_malacca | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.tb.chk_malacca.term_tw | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| east.tb.chk_malacca.term_kr | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| east.tb.chk_malacca.term_jp | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.ct.chk_malacca.fab_tw_leading_1 | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| east.ct.chk_malacca.fab_kr_memory_1 | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.ct.chk_malacca.osat_my | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| cape.ct.chk_cape.chk_malacca | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.ct.osat_my.chk_malacca | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.ct.osat_tw.chk_malacca | 12.15 | 4.28 | 4.15 | 0.12 | 6.00 | 13.00 | 26.00 | 38.96 | 1.10 |
| sea.tb.chk_taiwan.term_kr | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.tb.chk_taiwan.term_jp | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.chk_taiwan.fab_kr_memory_1 | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.chk_taiwan.osat_my | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.chk_taiwan.sink_cn | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.chk_taiwan.sink_jp | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.fab_kr_memory_1.chk_taiwan | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |
| sea.ct.fab_jp_memory_1.chk_taiwan | 8.75 | 2.67 | 2.50 | 0.09 | 5.00 | 12.00 | 22.00 | 33.15 | 1.10 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.98 | 7618 |
| 2 | 0.98 | 7352 |
| 3 | 0.95 | 7123 |
| 4 | 0.96 | 6648 |
| 6 | 0.96 | 5793 |
| 8 | 0.97 | 5052 |
| 12 | 0.92 | 4188 |
| 20 | 0.95 | 2361 |


### Added tariffs per edge (`graph_now.tariff`)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sea.ct.chk_taiwan.sink_cn | 51.40 | 37.38 | 37.20 | 0.53 | 21.00 | 48.00 | 52.00 | 71.04 | 0.20 |
| sea.ct.osat_tw.sink_cn | 51.40 | 37.38 | 37.20 | 0.53 | 21.00 | 48.00 | 52.00 | 71.04 | 0.20 |
| air.ct.osat_tw.sink_us | 28.60 | 19.09 | 19.25 | 0.29 | 17.00 | 40.00 | 52.00 | 64.72 | 0.10 |
| air.ct.osat_my.sink_us | 27.65 | 18.85 | 18.70 | 0.28 | 17.00 | 39.00 | 52.00 | 62.79 | 0.10 |
| air.ct.osat_kr.sink_us | 26.20 | 18.24 | 18.85 | 0.27 | 20.00 | 42.00 | 52.00 | 66.67 | 0.10 |
| lombok.tb.chk_hormuz.chk_taiwan | 19.65 | 12.65 | 14.05 | 0.20 | 16.00 | 34.00 | 52.00 | 60.05 | 0.10 |
| sea.ct.mat_ua_neon.chk_turkish | 19.65 | 13.54 | 13.10 | 0.20 | 19.75 | 39.50 | 52.00 | 66.50 | 0.10 |
| sea.ct.chk_suez.chk_malacca | 19.50 | 13.01 | 12.90 | 0.20 | 16.00 | 40.00 | 52.00 | 65.14 | 0.10 |
| cape.ct.chk_cape.chk_malacca | 19.50 | 13.01 | 12.90 | 0.20 | 16.00 | 40.00 | 52.00 | 65.14 | 0.10 |
| sea.ct.chk_malacca.chk_suez | 18.90 | 12.52 | 12.15 | 0.19 | 16.25 | 39.00 | 52.00 | 67.62 | 0.10 |
| sea.tb.chk_hormuz.chk_suez | 18.70 | 12.47 | 12.70 | 0.19 | 18.00 | 37.00 | 52.00 | 65.35 | 0.10 |
| cape.tb.chk_hormuz.chk_cape | 18.70 | 12.47 | 12.70 | 0.19 | 18.00 | 37.00 | 52.00 | 65.35 | 0.10 |
| sea.tb.src_au_lng.term_kr | 18.35 | 12.01 | 11.90 | 0.19 | 17.00 | 35.50 | 52.00 | 65.05 | 0.10 |
| sea.ct.mat_jp_wafer.fab_kr_memory_1 | 18.30 | 12.51 | 12.75 | 0.18 | 18.00 | 42.00 | 52.00 | 64.67 | 0.10 |
| air.ct.mat_jp_wafer.fab_kr_memory_1 | 18.30 | 12.51 | 12.75 | 0.18 | 18.00 | 42.00 | 52.00 | 64.67 | 0.10 |
| air.ct.mat_ua_neon.fab_tw_leading_1 | 18.30 | 12.35 | 12.55 | 0.18 | 19.00 | 40.00 | 52.00 | 64.85 | 0.10 |
| sea.ct.fab_kr_memory_1.chk_taiwan | 17.85 | 11.89 | 11.95 | 0.18 | 16.00 | 38.50 | 52.00 | 68.03 | 0.10 |
| air.ct.mat_ua_neon.fab_kr_memory_1 | 17.75 | 11.93 | 11.80 | 0.18 | 18.00 | 39.00 | 52.00 | 66.94 | 0.10 |
| pipe.tb.src_kz_uranium.grid_kr | 17.30 | 11.82 | 11.15 | 0.17 | 19.00 | 41.00 | 52.00 | 69.91 | 0.10 |
| sea.tb.chk_hormuz.chk_malacca | 17.25 | 11.46 | 11.50 | 0.17 | 18.00 | 37.00 | 52.00 | 66.09 | 0.10 |
| east.ct.fab_kr_memory_1.osat_my | 16.90 | 11.25 | 11.05 | 0.17 | 16.00 | 37.00 | 52.00 | 65.03 | 0.10 |
| air.ct.fab_kr_memory_1.osat_my | 16.90 | 11.25 | 11.05 | 0.17 | 16.00 | 37.00 | 52.00 | 65.03 | 0.10 |
| east.ct.fab_jp_memory_1.osat_my | 16.80 | 11.64 | 12.05 | 0.17 | 18.00 | 43.00 | 52.00 | 63.64 | 0.10 |
| air.ct.fab_jp_memory_1.osat_my | 16.80 | 11.64 | 12.05 | 0.17 | 18.00 | 43.00 | 52.00 | 63.64 | 0.10 |
| air.ct.osat_tw.sink_eu | 16.05 | 10.37 | 10.15 | 0.16 | 17.50 | 35.00 | 52.00 | 63.78 | 0.10 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.99 | 22544 |
| 2 | 0.99 | 22236 |
| 3 | 0.99 | 21897 |
| 4 | 0.99 | 21599 |
| 6 | 0.99 | 20955 |
| 8 | 0.99 | 20242 |
| 12 | 0.99 | 18813 |
| 20 | 0.99 | 16134 |


### Fabs (`graph_now.fab.R` < 1)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fab_tw_leading_1 | 4.15 | 2.48 | 2.20 | 0.04 | 16.00 | 32.00 | 46.50 | 61.45 | 0.95 |
| fab_tw_mature_1 | 4.15 | 2.48 | 2.20 | 0.04 | 16.00 | 32.00 | 46.50 | 61.45 | 0.95 |
| fab_jp_memory_1 | 2.05 | 1.24 | 1.15 | 0.02 | 16.00 | 31.00 | 49.00 | 66.67 | 0.94 |
| fab_kr_memory_1 | 1.95 | 1.16 | 1.10 | 0.02 | 17.50 | 29.00 | 49.00 | 56.41 | 0.93 |
| fab_eu_leading_1 | 1.90 | 1.30 | 1.25 | 0.02 | 15.50 | 40.00 | 52.00 | 70.00 | 0.94 |
| fab_eu_mature_1 | 1.90 | 1.30 | 1.25 | 0.02 | 15.50 | 40.00 | 52.00 | 70.00 | 0.94 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 1.00 | 327 |
| 2 | 0.99 | 324 |
| 3 | 0.99 | 321 |
| 4 | 1.00 | 317 |
| 6 | 0.99 | 313 |
| 8 | 1.00 | 302 |
| 12 | 1.00 | 281 |
| 20 | 0.98 | 226 |


### OSATs (`graph_now.osat.R` < 1)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| osat_tw | 4.15 | 2.48 | 2.20 | 0.04 | 16.00 | 32.00 | 46.50 | 61.45 | 0.95 |
| osat_kr | 1.95 | 1.16 | 1.10 | 0.02 | 17.50 | 29.00 | 49.00 | 56.41 | 0.93 |
| osat_my | 0.90 | 0.53 | 0.70 | 0.01 | 19.50 | 34.00 | 42.75 | 38.89 | 0.99 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.99 | 140 |
| 2 | 1.00 | 137 |
| 3 | 0.99 | 137 |
| 4 | 1.00 | 134 |
| 6 | 1.00 | 132 |
| 8 | 1.00 | 129 |
| 12 | 0.99 | 122 |
| 20 | 0.98 | 97 |


### Grids (`graph_now.grid.G_bar` < nominal)

| unit | episodes hit % | weeks hit % | hit in week 1 % | spells / episode | weeks p25 | p50 | p75 | runs to the end % | median level when hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_tw | 22.05 | 2.32 | 2.40 | 0.25 | 2.00 | 3.00 | 6.00 | 10.14 | 0.74 |
| grid_kr | 21.15 | 2.39 | 2.40 | 0.24 | 2.00 | 4.00 | 6.25 | 8.40 | 0.76 |
| grid_jp | 20.40 | 2.25 | 2.10 | 0.23 | 2.00 | 4.00 | 6.00 | 7.28 | 0.73 |
| grid_eu | 19.25 | 2.03 | 2.45 | 0.22 | 2.00 | 4.00 | 6.00 | 8.97 | 0.74 |

Persistence, all units pooled:

| hit for d weeks | P(still hit next week) | spells |
| --- | --- | --- |
| 1 | 0.91 | 1892 |
| 2 | 0.77 | 1703 |
| 3 | 0.76 | 1293 |
| 4 | 0.76 | 960 |
| 6 | 0.81 | 537 |
| 8 | 0.79 | 332 |
| 12 | 0.85 | 133 |
| 20 | 0.91 | 34 |


## 3. Messages: how often a thread is real, and how much time it gives

A thread is one event's announcements. A decoy is withdrawn at the moment it would have taken effect, so
until then it looks like a real one. Only threads whose effect falls inside the episode are ever shown.
The regime publishes its decoy shares in `config['regime']['phi']`: tariff_formal 0.25, tariff_informal 0.36, ties_threat 0.538, mid_threat 0.298.

| first message | threads / episode | decoys % | real: weeks to effect p10 | p25 | p50 | p75 | p90 | decoy: weeks to withdrawal p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tariff_formal | 2.56 | 25.99 | 2.00 | 5.00 | 9.00 | 15.00 | 33.00 | 9.00 |
| tariff_informal | 2.31 | 35.97 | 0.00 | 1.00 | 3.00 | 7.00 | 14.00 | 3.00 |
| sanction_legal | 1.95 | 13.91 | 0.00 | 0.00 | 0.00 | 0.00 | 4.00 | 5.00 |
| ties_threat | 3.71 | 48.41 | 0.00 | 0.00 | 13.00 | 57.00 | 100.00 | 24.00 |
| mid_threat | 0.61 | 25.20 | 0.00 | 0.00 | 1.00 | 3.00 | 9.00 | 2.00 |

### P(real) of a thread still open a weeks after its first message

| first message | a = 0 % | a = 1 % | a = 2 % | a = 4 % | a = 8 % | a = 13 % | a = 26 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tariff_formal | 74.01 | 73.98 | 73.88 | 73.83 | 73.50 | 73.16 | 71.29 |
| tariff_informal | 64.03 | 63.76 | 63.65 | 64.45 | 64.47 | 65.02 | 66.49 |
| sanction_legal | 48.72 | 48.56 | 48.63 | 48.84 | 46.27 | 44.83 | - |
| ties_threat | 45.82 | 46.29 | 46.53 | 46.98 | 47.29 | 47.32 | 46.81 |
| mid_threat | 71.74 | 71.99 | 67.74 | 65.00 | 62.35 | 64.95 | - |

### Follow-up messages (a dated notice after the first message)

| follow-up | threads that get one % | decoys among them % | threads opening with it / episode | decoys among those % | real: weeks from it to effect p25 | p50 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sanction: legal publication | 54.80 | 5.86 | 1.95 | 13.91 | 0.00 | 0.00 | 0.00 |
| tariff: final notice | 98.67 | 29.79 | 0.00 | 0.00 | 0.00 | 1.00 | 2.00 |

### Pending prohibitions (`pending_prohibitions.*`)

An entry is a sanction's legal publication dated at least a week ahead. Most real publications come out
the week the sanction starts, so they never show up here.

| weeks ahead | publications / episode | decoys % |
| --- | --- | --- |
| all | 0.73 | 51.95 |
| 1 week | 0.03 | 43.28 |
| 2-3 weeks | 0.10 | 57.58 |
| 4-8 weeks | 0.40 | 51.07 |
| 9+ | 0.20 | 52.35 |

Weeks ahead of a real one: p25 4, p50 5, p75 9, p90 13.

### Share of new events an agent hears of in advance

| type | new / episode | announced >= 1 week ahead % | >= 4 weeks ahead % |
| --- | --- | --- | --- |
| tariff | 3.38 | 93.70 | 69.63 |
| sanction | 3.59 | 47.01 | 38.49 |
| material_outage | 0.21 | 0.00 | 0.00 |
| militarised_closure | 0.46 | 65.43 | 24.62 |
| regional_conflict | 0.27 | 0.00 | 0.00 |
| piracy | 0.28 | 0.00 | 0.00 |
| energy_shock | 1.35 | 0.00 | 0.00 |
| weather_closure | 1.42 | 0.00 | 0.00 |
| port_strike | 1.09 | 0.00 | 0.00 |

## 4. Warning scores: what a high score is worth

`warning.score` is 0.604 x the unit's latent risk of 1 week(s) ago plus noise, so
it moves slowly and never names an event. Each cell is the share of unit-weeks with that score in which the
thing happens within the next h weeks; the last row is the AUROC of the score for it (0.5: no information).

### Straits: any disruption, while the strait is open

| score | share of unit-weeks % | disrupted within 1 w % | disrupted within 4 w % | disrupted within 8 w % | disrupted within 13 w % |
| --- | --- | --- | --- | --- | --- |
| -inf to -1 | 15.94 | 0.44 | 1.71 | 3.25 | 5.05 |
| -1 to 0 | 33.99 | 0.44 | 1.74 | 3.28 | 5.01 |
| 0 to 1 | 34.32 | 0.46 | 1.79 | 3.45 | 5.26 |
| 1 to 2 | 13.53 | 0.52 | 1.95 | 3.70 | 5.62 |
| 2 to inf | 2.22 | 0.58 | 2.02 | 3.69 | 5.36 |
| AUROC |  | 0.52 | 0.51 | 0.51 | 0.51 |

### Straits: a militarised closure starts

| score | share of unit-weeks % | starts within 1 w % | starts within 4 w % | starts within 8 w % | starts within 13 w % |
| --- | --- | --- | --- | --- | --- |
| -inf to -1 | 15.94 | 0.05 | 0.21 | 0.43 | 0.71 |
| -1 to 0 | 33.99 | 0.07 | 0.25 | 0.46 | 0.70 |
| 0 to 1 | 34.32 | 0.08 | 0.34 | 0.66 | 0.99 |
| 1 to 2 | 13.53 | 0.14 | 0.48 | 0.88 | 1.32 |
| 2 to inf | 2.22 | 0.16 | 0.66 | 1.25 | 1.70 |
| AUROC |  | 0.59 | 0.59 | 0.58 | 0.58 |

| strait | disrupted within 13 w: any score % | score >= 1 % | AUROC | militarised closure within 13 w: any score % | score >= 1 % | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| chk_hormuz | 5.28 | 6.03 | 0.52 | 1.19 | 2.04 | 0.59 |
| chk_malacca | 6.50 | 6.89 | 0.51 | 2.36 | 3.08 | 0.54 |
| chk_suez | 5.55 | 4.96 | 0.50 | 1.13 | 1.56 | 0.61 |
| chk_cape | 4.48 | 4.35 | 0.49 | 0.00 | 0.00 | - |
| chk_taiwan | 6.71 | 8.19 | 0.54 | 2.35 | 4.09 | 0.60 |
| chk_panama | 4.21 | 5.78 | 0.53 | 0.00 | 0.00 | - |
| chk_turkish | 4.27 | 3.85 | 0.50 | 0.00 | 0.00 | - |

### Regions: the region enters a conflict state

| score | share of unit-weeks % | in conflict within 1 w % | in conflict within 4 w % | in conflict within 8 w % | in conflict within 13 w % |
| --- | --- | --- | --- | --- | --- |
| -inf to -1 | 18.52 | 0.00 | 0.06 | 0.14 | 0.23 |
| -1 to 0 | 36.08 | 0.00 | 0.14 | 0.32 | 0.51 |
| 0 to 1 | 32.41 | 0.00 | 0.30 | 0.68 | 1.10 |
| 1 to 2 | 11.47 | 0.00 | 0.55 | 1.23 | 1.98 |
| 2 to inf | 1.52 | 0.00 | 0.92 | 2.15 | 3.49 |
| AUROC |  | - | 0.70 | 0.70 | 0.70 |

### Regions: militarised_closure, regional_conflict, piracy, energy_shock

| score | share of unit-weeks % | new event within 1 w % | new event within 4 w % | new event within 8 w % | new event within 13 w % |
| --- | --- | --- | --- | --- | --- |
| -inf to -1 | 16.42 | 0.26 | 0.82 | 1.48 | 2.20 |
| -1 to 0 | 33.94 | 0.27 | 0.87 | 1.54 | 2.26 |
| 0 to 1 | 33.78 | 0.26 | 0.87 | 1.56 | 2.34 |
| 1 to 2 | 13.54 | 0.32 | 1.04 | 1.84 | 2.67 |
| 2 to inf | 2.31 | 0.39 | 1.28 | 2.17 | 3.08 |
| AUROC |  | 0.52 | 0.52 | 0.52 | 0.52 |

### Regions: tariff, sanction, material_outage

| score | share of unit-weeks % | new event within 1 w % | new event within 4 w % | new event within 8 w % | new event within 13 w % |
| --- | --- | --- | --- | --- | --- |
| -inf to -1 | 16.42 | 0.83 | 2.79 | 4.97 | 7.32 |
| -1 to 0 | 33.94 | 0.85 | 2.82 | 5.07 | 7.50 |
| 0 to 1 | 33.78 | 0.88 | 2.92 | 5.23 | 7.69 |
| 1 to 2 | 13.54 | 0.91 | 3.04 | 5.44 | 7.98 |
| 2 to inf | 2.31 | 0.97 | 3.14 | 5.57 | 8.12 |
| AUROC |  | 0.51 | 0.51 | 0.51 | 0.51 |
