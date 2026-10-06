# cap: a reactive cap on a fab's wafer orders

Agent: `MAIN/outputs/heur4/agents/cap/` (only `chip_part.py` differs from v2; diff in `cap.diff`). Scratch, rows and logs: `MAIN/outputs/heur4/cap/`.

## The overflow signal (exact)
Week t-1 closes with I(t-1) = min(storage, I(t-2) + lots finished - raw chips shipped) (raw chips leave a fab only by shipping). Disposal(t-1) = I(t-2) + finished - shipped - I(t-1) when the store ends at its storage, else 0. Finished = wip seen at the last observation with `out_week == t-1`; shipped = `last_week.clip.executed` summed over the fab's raw slots; I(t-2) is kept from the last observation.
Check against the simulator's `disposal` record (scratch/sigcheck.py): Small root 111 ep 3 and Full root 111 ep 0, every fab: estimated = true disposal to 1e-9 relative (Small 218,252 = 218,252; Full 4,272,231 = 4,272,231), abs error 0 in every week after the reset window. Reset window = last `out_week` in the first observation's wip (8 weeks): disposals up to that week are not used.

## The rule
In `_wafer_flows`, after `want` is computed: with `over` = mean disposal of the fab over the last `cap_win` weeks (reset weeks excluded), if over > 0:
`want = min(want, max(want - cap_gain * over, cap_floor * sent_rate))` where `sent_rate` = mean raw chips shipped by the fab over the last `cap_sent_win` weeks. Condition is "store full and disposing", not a level. Recovery: when disposal stops the window empties after `cap_win` weeks and the order is the old one again; the floor keeps the fab's wafers at 0.8 of what it ships, so outlets stay busy and a cut fab is never starved. The fuel part is untouched (reads the cut wafer entries; chip part runs first).
Functions: `_overflow_update` (new, balance + history), `overflow_rate`, `sent_rate`, `_cap_applies` (chip rank / shed filters, off by default) new; `_wafer_flows` (call + 5 lines), `__init__` (4 history fields), `PARAMS`.
Switch: `PARAMS["cap"]` (False = v2: Small x64 cost identical to base, +0.0000). Numbers (delivered defaults): `cap` True, `cap_gain` 16, `cap_win` 6, `cap_floor` 0.8, `cap_sent_win` 6, `cap_chips` [], `cap_shed_n` 0 (shed filter off), `cap_shed_win` 13. Each can be overridden by params.json.

## Every variant (costs.py, root 111, vs v2; Full x32 base 0.8465, Small x64 base 0.8104)
Names: win/gain/floor. Interval is 90%.
| variant | Full x32 | Small x64 |
|---|---|---|
| g1 w6 floor0 | +0.0003 (+0.0002..+0.0004) | |
| g2 w6 | +0.0006 (+0.0004..+0.0008) | +0.0005 (+0.0001..+0.0009) |
| g4 w6 | +0.0012 (+0.0007..+0.0016) | -0.0007 (-0.0021..+0.0004) |
| g8 w6 | +0.0019 (+0.0008..+0.0030) | -0.0017 (-0.0039..+0.0001) |
| g16 w6 | +0.0019 (+0.0002..+0.0036) | |
| w3 g1 | +0.0004 (+0.0003..+0.0005) | +0.0002 (-0.0001..+0.0005) |
| w3 g2 | +0.0006 (+0.0004..+0.0008) | +0.0005 (+0.0002..+0.0009) |
| w3 g8 | +0.0021 (+0.0011..+0.0031) | -0.0006 (-0.0018..+0.0004) |
| w3 g16 | +0.0023 (+0.0007..+0.0039) | -0.0017 (-0.0038..-0.0001) |
| w3 g64 | +0.0013 (-0.0003..+0.0030) | |
| w4 g32 | +0.0012 (-0.0002..+0.0027) | |
| w1 g64 | +0.0018 (+0.0007..+0.0030) | |
| w2 g8 | +0.0019 (+0.0010..+0.0028) | |
| w12 g2 | +0.0005 (+0.0004..+0.0007) | |
| w12 g8 | +0.0016 (+0.0007..+0.0025) | |
| w3 g8 dearest chip only (`cap_chips` [0]) | +0.0016 (+0.0007..+0.0025) | -0.0007 (-0.0018..+0.0003) |
| w3 g8 cheaper chip only ([1]) | +0.0004 (+0.0001..+0.0007) | |
| w3 g8 only grids with >=4 shed weeks in 13 | +0.0004 (-0.0001..+0.0009) | |
| w3 g16 floor 1.0 | +0.0018 (+0.0009..+0.0027) | |
| w3 g16 floor 0.8 | +0.0021 (+0.0010..+0.0031) | +0.0005 (+0.0001..+0.0010) |
| w3 g64 floor 1.0 | +0.0020 (+0.0010..+0.0030) | +0.0003 (-0.0003..+0.0010) |
| w3 g64 floor 1.2 | +0.0018 (+0.0008..+0.0028) | |
| w3 g64 floor 0.6 | +0.0025 (+0.0012..+0.0038) | -0.0001 (-0.0009..+0.0007) |
| w6 g16 floor 0.8 (DELIVERED) | **+0.0026 (+0.0014..+0.0038)** | **+0.0003 (-0.0002..+0.0009)** |
| w3 g16 floor 0.8, sent_win 3 | +0.0022 (+0.0012..+0.0033) | +0.0006 (+0.0002..+0.0012) |
| w3 g16 floor 0.8, sent_win 12 | +0.0022 (+0.0012..+0.0032) | |
| w3 g8 floor 0.7 | +0.0019 (+0.0011..+0.0028) | |
Without a floor, gain above 2 helps Full and hurts Small (over-cut starves outlets); the floor (wafers never below 0.6 to 1.2 of the shipping rate) removes that: all floored variants are +0.0018 to +0.0026 on Full and -0.0001 to +0.0006 on Small. Differences among floored variants are within noise (the delivered one was picked among ~10, the interval does not count that choice). "Only at grids that shed" loses most of the gain (+0.0004): the cut also saves wafer and disposal fee at complete grids. The dearest chip carries about 3/4 of the gain.

## account.py, root 111, 16 episodes, per episode (v2 -> cap)
| | Full | Small |
|---|---|---|
| total J, bn | 6182.9 -> 6173.2 | 2783.2 -> 2783.4 |
| lots started (sum over fabs), M | 80.68 -> 75.05 (-5.6) | 9.95 -> 9.42 |
| raw chips disposed at fabs, M | 11.27 -> 6.12 | 2.17 -> 1.75 |
| disposal cost, bn | 20.0 -> 13.4 | 3.5 -> 2.9 |
| fab energy, GWh | 97,575 -> 90,375 (-7.2 TWh) | 13,821 -> 13,185 |
| shed cost, bn | 3948.9 -> 3944.3 | 1861.0 -> 1860.3 |
| shortage cost, bn | 2155.8 -> 2157.7 | 906.2 -> 907.9 |
| raw chips shipped, M | 80.40 -> 80.04 | 13.72 -> 13.65 |
Chips sold: demand served per week changes by 0.0 to -0.3 thousand per sink on both boards (e.g. Full sink_us chip_le 87.8 -> 87.8, chip_mat 103.1 -> 102.9, sink_jp chip_mat 31.9 -> 31.8): a slight loss, shortage +1.9 bn Full, +1.7 bn Small. Full saves 9.7 bn net (shed -4.6, disposal -6.6, shortage +1.9). On Small the saving in disposal and shed is eaten by shortage (net 0).
Full v2 vs cap raw chips disposed: still 6.1 M of 11.3 M left (the delay of 9 weeks and the 8 reset weeks).

## sbf check (cap, in this checkout, no docker)
Small: week 1 0.011 s, median 0.0031 s, max 0.011 s (budget 2). Full: week 1 0.038 s, median 0.0127 s, max 0.038 s (budget 4). Both pass.

## What I would try next
- A forecast instead of a trailing mean: project the store with lots in process (known out weeks) and the fab's shipping rate for weeks t+9.., and cut by the projected overflow of the arrival week.
- Cut the wafer target of an overflowing fab (it holds 3 weeks of wafers that start anyway) and shorten that buffer.
- Floor tied to the plants' outlet capacity (sale_rates) instead of the trailing shipped rate; per-fab gain.
- Use 64 episodes on Full for the delivered variant; I only confirmed x32.

## Brief notes
- Brief says v2 Small 0.826 / Full 0.834; on root 111 x64 / x32 the base is 0.8104 / 0.8465 (different set), differences only are comparable.
- The brief's exploration on Full x16 was skipped: Full x32 takes about 55 s here.
