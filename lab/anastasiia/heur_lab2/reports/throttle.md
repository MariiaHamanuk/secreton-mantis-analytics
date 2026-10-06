# throttle: stop a grid from offering power that nobody takes

Agent: `outputs/heur2/agents/throttle/` (`agent.py`, `fuel_part.py`, `chip_part.py`; `chip_part.py` is byte-identical to the
base). Worktree with the scripts, the arrays and every variant folder (each with its `params.json`):
`/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.claude/worktrees/agent-a65cee051e9dd7b82/lab/throttle/`.

## Result

| set | base | throttle | difference (90% interval) |
| --- | --- | --- | --- |
| Small, root 111, 64 episodes | 0.7821 | 0.7932 | **+0.0111 (+0.0089 to +0.0135)** |
| Small, root 111, 64 episodes, `top_weeks` 8 on both sides | 0.7817 | 0.7927 | +0.0110 (+0.0087 to +0.0134) |
| Full, root 111, 16 episodes | 0.7705 | 0.7739 | +0.0034 (+0.0015 to +0.0058) |
| Full, root 111, 16 episodes, `top_weeks` 8 on both sides | 0.7706 | 0.7732 | +0.0026 (+0.0011 to +0.0046) |
| Small, root 444, 40 episodes (diagnostic): J per episode | 2,734.9 bn | 2,723.8 bn | -11.1 bn (about +0.010); better in 37 of 40 episodes |

`sbf check` passes on Tiny, Small and Full (longest week 0.011 s on Small, 0.035 s on Full). Two runs of the same episode give
identical weekly traces. Gain by harm level (Small, 64 episodes): 0.736 -> 0.742, 0.824 -> 0.835, 0.832 -> 0.852,
0.802 -> 0.825: it grows with the level, because the fuel is scarce in the harsh episodes.

## The rule, in plain words

A grid offers `g_av` = free segment + every fuel's output (the rationed fuel's is `share x G_bar x min(1, last week's closing
stock / threshold)`, never more than the fuel on hand). The simulator serves the base load first, then the fabs' ask, and runs
every segment at `load = (y + fab energy) / g_av`. An offer above the load therefore burns fuel in place of the free
segment. Each week the fuel part now computes, for NEXT week:

1. what each fab will ask: `e x min(alpha x cap0, wafers on hand / R)`, where the wafers next week = on hand now + arriving
   now - started now (this needs this week's offer) + arriving next week (the pipeline's last-leg entries + this week's wafer
   flows with a one-week lead, **as the simulator will clip them**: edge capacity, then the source's stock, pro rata);
2. the load: base load + that ask (the ask counts only if a lot started next week can still be sold: `t + 1 + lag <= T`,
   `lag` = tau + 1 + raw route lead + plant tau + 1 + packaged route lead; if less than `thr_useful` of the ask is for lots that
   can still be sold, the fabs do not count at all);
3. `rho` = (load - free segment - what the other fuels will offer) / (rationed fuel's full output), clipped to [0, 1];
4. the valve closes the week at `rho x threshold` (not `threshold + ss_weeks x burn`), counting the burn the grid will really
   make this week (`av x load`, not `av`).

The hold / prime / run decision (`_hold`) works on the same quantities: threshold `rho x thr` and weekly burn `rho x b_nom`; a
prime week moves what brings the grid to the level AFTER its own burn (the base's prime formula ignores the burn).
Lead's rule (point 2 of the note): when the terminals of the grid and fuel hold more than `thr_full` = 0.6 of their storage, the
closing level is the base level (`thr + ss_weeks x b_nom`), everything else (the burn, the `_hold` quantities) stays as above.

Switches (in `FUEL`, all keep the base behaviour when off): `throttle`, `thr_endgame`, `thr_useful`, `thr_margin`, `thr_floor`,
`thr_hold`, `thr_min`, `thr_full`, `thr_prime`, `thr_clip`.

## Functions touched (the lead merges by hand)

`agent.py`: `Agent.act` calls `fill_chip_flows` first and `fuel.fill` second (the two fill disjoint slots).

`fuel_part.py` (line numbers of the final file):
- `FUEL` dict, lines 78-88 (new keys).
- `FuelRules.__init__`, lines 210-254 (new block: `null_share`, `base_load`, `fabs`, `grid_fabs`, `wafer_sl`; read from `config` only).
- `_hold`, line 289: signature gains `end=None`; lines 304-306: docstring; lines 322-325: two new branches in front of the
  base's `return max(0.0, thr + ss x b_nom - s0 - landed), "prime"` (now line 326).
- New methods `_fab_state`, `_executed`, `_throttle` (lines 329-446, between `_hold` and `_lots`).
- `fill` (def at line 467, signature unchanged): docstring lines 468-471; line 489 (`arrive_next, wafer_in`); lines 501-503
  (wafers on their last leg, in the pipeline loop, inside `if int(k) not in self.fuel_set:` before its `continue`); lines
  511-512 (`elif ... t + 1: arrive_next[...]`); lines 574-575 (`y_bar`, `fstate`); lines 579-582 (`th = None` and the
  `_throttle` call); lines 594-598 (the throttled `end` and `burn`, with the `thr_full` rule; line 593 above them is the base's);
  line 599 (the `plan[(g, k)]` dict gains `"th": th`); lines 604-611 (the `_hold` call: throttled arguments `th["thr"]`,
  `th["b"]` and `end` when `th` exists, else the base call); lines 703-707 (trace). Nothing in the order passes, the crude
  logic or the `bank` branch changed.

## Every variant (Small, root 111, 32 episodes, paired against the base 0.7843 unless said)

| variant | what | score | difference | 90% interval |
| --- | --- | --- | --- | --- |
| V1 | first version: rho from the forecast ask, throttled `_hold`, endgame rule; wafers next week from the pipeline only, requests taken as asked, prime week ignoring the burn | 0.7918 | +0.0075 | +0.0055 to +0.0096 |
| V2 | V1 + wafer requests as the simulator clips them + prime week counts the burn + the chip part's picture of cargo at straits | 0.7943 | +0.0100 | +0.0071 to +0.0130 |
| noprime | V2 with the base's prime formula (`thr_prime` off) | 0.7902 | +0.0059 | -0.0025 to +0.0121 |
| noclip | V2 without the clip emulation (`thr_clip` off) | 0.7918 | +0.0075 | +0.0055 to +0.0096 |
| notransit | V2 without the chip part's cargo-at-straits picture (the final code) | 0.7941 | +0.0099 | +0.0066 to +0.0131 |
| nohold | V2 with `_hold` on the unthrottled threshold and burn (`thr_hold` off) | 0.7879 | +0.0037 | -0.0009 to +0.0081 |
| noend | V2 without the endgame rule (`thr_endgame` off) | 0.7938 | +0.0095 | +0.0067 to +0.0124 |
| pulse0 | V2 with no time concentration (`pulse` off) | 0.6985 | -0.0857 | -0.1086 to -0.0617 |
| m10 | `thr_margin` 0.1 (offer 10% above the ask) | 0.7945 | +0.0102 | +0.0075 to +0.0131 |
| f10 | `thr_floor` 0.1 (a tenth of a week of burn above the level) | 0.7853 | +0.0010 | -0.0006 to +0.0025 |
| use20 | `thr_useful` 0.2 (the ask counts when a fifth of it is for lots that can be sold) | 0.7942 | +0.0100 | +0.0071 to +0.0129 |
| orat100 | `on_ratio` 1.0 | 0.7938 | +0.0096 | +0.0062 to +0.0131 |
| band80 / band95 | `run_band` 0.8 / 0.95 | 0.7851 / 0.7945 | +0.0008 / +0.0102 | -0.0104 to +0.0095 / +0.0067 to +0.0139 |
| prime1 / prime3 | `prime_weeks` 1 / 3 | 0.7680 / 0.7908 | -0.0163 / +0.0065 | -0.0355 to +0.0004 / +0.0034 to +0.0096 |
| end14 / end12 / end18 | `end_weeks` 14 / 12 / 18 | 0.7931 / 0.7904 / 0.7921 | +0.0089 / +0.0061 / +0.0078 | +0.0055 to +0.0121 / +0.0017 to +0.0104 / +0.0039 to +0.0117 |
| cap95 | V2 + orders of the rationed fuel never fill the terminals past 95% of their storage | 0.7944 | +0.0101 | +0.0072 to +0.0131 |
| pask05 / pask10 / pask20 | V2 + no pulsing while the fabs' smoothed ask is below 5 / 10 / 20% of the sliver | 0.7941 / 0.7940 / 0.7938 | +0.0099 / +0.0098 / +0.0096 | +0.0070 to +0.0129 / +0.0069 to +0.0127 / +0.0068 to +0.0125 |
| full60-v1 | `th = None` whenever the terminals are over 60% full (the first reading of the lead's rule) | 0.7941 | +0.0098 | +0.0064 to +0.0135 |
| full60-v1, 64 episodes | the same on 64 episodes (base 0.7821) | 0.7895 | +0.0074 | +0.0023 to +0.0117 |
| final, 64 episodes | the corrected rule: only the closing level reverts to the base level | 0.7932 | +0.0111 | +0.0089 to +0.0135 |
| base-primefix, 64 episodes | the base with only the prime week counting the burn | 0.7822 | +0.0000 | -0.0002 to +0.0003 |
| Full 16: V2 | V2 (no `thr_full`) | 0.7718 | +0.0013 | -0.0012 to +0.0042 |
| Full 32: V2 | V2 (no `thr_full`), 32 episodes | 0.7890 | +0.0003 | -0.0012 to +0.0017 |
| Full 16: final | with `thr_full` 0.6 | 0.7739 | +0.0034 | +0.0015 to +0.0058 |
| Full 16, `top_weeks` 8: final / no `thr_full` | against the base with `top_weeks` 8 | 0.7732 / 0.7714 | +0.0026 / +0.0007 | +0.0011 to +0.0046 / -0.0014 to +0.0033 |

## What moved (`account.py`)

Root 444, 40 episodes, GWh a week of free segment lost (weeks 1-38 / 39-52):

| grid | base | throttle without `thr_full` | final (`thr_full` 0.6) |
| --- | --- | --- | --- |
| TW | 54.5 / 58.5 | 1.2 / 0.0 | 23.9 / 34.4 |
| KR | 38.2 / 60.5 | 0.7 / 0.0 | 15.3 / 48.9 |
| JP | 3.7 / 7.4 | 0.8 / 0.0 | 1.2 / 2.4 |
| EU | 16.0 / 10.9 | 1.0 / 0.0 | 0.8 / 1.8 |
| per episode | 6,193 | 141 | 2,793 |

The score is better with the 2,793 than with the 141 (J 2,723.8 against 2,724.4 on root 444; Full +0.0034 against +0.0012):
fuel saved is worth something only where the grid is short of that fuel at other times, and where the terminals are full the
saved fuel is thrown away (the lead's point 1). Gap to the best plan on root 444 (final): total 141.6 -> 130.4 = chips 66.3 ->
67.6, shed 68.0 -> 55.6, other 7.3 -> 7.2; shed gap by grid TW 12.3 -> 8.1, KR 23.0 -> 18.3, JP 10.9 -> 10.3, EU 21.8 -> 18.9.
Energy to the fabs 13,813 -> 13,614 GWh (no lots in the last weeks).

Root 111, 16 episodes, throttle without `thr_full` (V2): free lost TW 65.7 -> 1.3, KR 40.6 -> 0.6, JP 4.3 -> 0.3, EU 13.1 -> 0.8;
shed (GWh a week, weeks 1-38) TW 359 -> 311, KR 945 -> 912, JP 2974 -> 2967, EU 4205 -> 4200; lng thrown away at terminals per
episode TW 4,116 -> 6,786, KR 3,856 -> 6,061, EU 7,482 -> 8,128. In 8 of 16 episodes one terminal (TW in 5, KR in 3, EU in 2)
stays at 100% of its storage for 25-30 weeks: there the saved fuel is disposed of and nothing is gained.

Predictions against the simulator (8 episodes of root 111, V2): the ration the simulator uses next week is within 0.02 of the
target in 98% (TW), 99% (KR), 100% (JP, EU) of the weeks. The ask forecast: mean absolute error TW 8.3 GWh, KR 6.1, JP 0.0, EU 1.5;
the real ask exceeds the forecast by more than 10 GWh in 28% (TW) and 22% (KR) of the weeks (cargo released from a strait queue).

## What mattered

1. The valve's level at `rho x threshold` with the burn the grid will really make, from the forecast ask: V1 +0.0075.
2. `_hold` on the throttled threshold and burn, with a prime week that counts its own burn: without the throttled `_hold` +0.0037,
   without the prime fix +0.0059 (and the `thr_full` rule makes the prime fix necessary, see below).
3. Wafer requests as the simulator will clip them: +0.0025 (an episode where the supplier's stock was empty: the chip part asked
   for 313 GWh worth of wafers each week for 40 weeks, nothing came, the grid wasted 183 GWh a week).
4. `thr_full` 0.6: +0.0009 on Small, +0.002 on Full.
Small: the endgame rule +0.0005, the cargo-at-straits picture 0, a margin 0, a floor negative, order cap 0, no pulsing when the ask
is small 0.

## What was wrong or incomplete in the README and in the brief

- "6,193 GWh, 25.5 bn, about 0.024": the 6,193 reproduces exactly (root 444), but 1,922 of it (31%) is in the last 14 weeks,
  where a grid has no use for saved fuel (the endgame rule is worth 0.0005), and where the terminal is full the saved fuel is
  disposed of. The measured gain is 0.011 (11 bn on root 444).
- "close the week at `rho* x threshold` plus a small margin": the margin is a loss. A tenth of a week of burn kills the gain (f10);
  `thr_margin` is neutral. The ration reaches its target in 98 to 100% of the weeks.
- "Wafers next week from `pipeline.*`, `stock.qty` and this week's wafer flows": true, but the chip part's requests are not the
  executed flows (it does not look at the source's stock), and cargo released from a strait queue arrives a week later than the
  pipeline says. The clip emulation is worth +0.0025.
- "no pulsing when the fabs will not ask": with the throttle a complete week burns what a steady ration burns, and shed is linear in
  VOLL, so a pulse with no ask costs nothing and gains nothing; not pulsing gives 0.
- The base's prime week moves `thr + ss x b_nom - s0 - landed` and ignores what the grid burns that week (closing = target - burn).
  Alone this changes nothing (base-primefix +0.0000: prime weeks start from an empty grid), but with partial stocks it
  produces loops: in root 111 episode 47 (KR, `th = None` on a full terminal) the grid sat in "prime" for 16 weeks at ration 0.62
  (shed 850 GWh a week, +121 bn). Counting the burn in the prime week removes it.
- On Full the fabs ask the full sliver at the big grids (TW 488 of 488.5 GWh a week) and the gas-rich grids (US: no terminal-to-grid
  slot, lng arrives at the grid itself; IN: no fabs) have nothing to gain, so the gain is small there.

## What I would try next

- Exact FIFO emulation of the strait releases (cohorts by arrival week, kappa, next-edge capacity) for next week's wafers; the
  forecast is short by 7 GWh a week at TW. Margin and the chip part's picture did nothing, so the gain is probably below 0.001.
- Chips are 1.3 bn worse on root 444 (66.3 -> 67.6 after the 60% rule; 1.8 before it): lots in the last two useful weeks at JP (a
  grid that held its stock for two complete weeks in the base only by the prime formula's accident: root 444 episodes 19 and 35)
  and at TW's mature fab in weeks 39-40 (its lots are still sold, but its share of TW's ask is 28%, under `thr_useful`, so the
  grid offers the base load only; `thr_useful` 0.2 changed nothing in the score, the sliver is shared pro rata with the leading
  fab's useless lots).
- The valve of the rationed fuel at grids without a terminal slot (lng arrives at the grid).
- With fewer wafers (the other branch) the ask falls and `rho` falls with it: re-measure `end_weeks` and `prime_weeks` then.
