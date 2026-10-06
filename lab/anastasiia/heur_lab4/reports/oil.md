# oil: fuel into the wrong grid (shared first edge)

Verdict: the hypothesis does not hold. No agent change was made (oil.diff is empty, no folder in outputs/heur4/agents/oil).
Scripts: outputs/heur4/oil/tools/ (diag2.py, d3..d9.py, small.py). Data: v2 Full root 444, 7 episodes matched to the plan (episode 3 skipped); Small root 444, 40 episodes.

## 1. Gulf first edge (Full, v2; "short" = grid sheds and its crude segment is below its cap, last week excluded)
- Edge sea.tb.src_gulf_crude.chk_hormuz (true cap 3,160 a week): v2 sends 215k per episode against 329k of capacity (65%). The plan ships 185k there, i.e. 30k LESS than v2 and burns more at KR/JP/CN.
- In the weeks a grid is short of crude the edge is 57% (JP, CN, KR) to 66% (SEA) full; at or above 97% full in 18% of JP's short weeks, 7% of CN's, 26% of KR's, 25% of SEA's. Mean slack 1,300 a week. Edge full in 23 weeks per episode, and in 20.7 of them some of JP/CN/SEA/KR is short.
- Share to IN in short weeks: about 550 of 1,800 a week. IN in total: 56.4k sent, 36.9k burned, 24.5k disposed at term_in. In edge-full short weeks IN sends 7.6k per episode against a burn of 7.3k: at most 2.8k per episode could go elsewhere (about 11 bn if all burned in short weeks, 0.003 of score), and the Hormuz queue is empty then (14) while Malacca+Taiwan hold 1.3k, so the cargo would not clearly get through.
- Order of service already puts IN last (no fabs): its top-up takes what the others' targets left. Only its essential level (small) precedes others' top-ups.
- The crude is not missing in the short weeks, it is at the terminal: opening terminal stock in short weeks SEA 4,400 (cap 749 a week), CN 2,460 (cap 1,797), JP 990 (cap 889), KR 227 (cap 215). The deficit physically on hand at start of week: SEA 41k of 55k per episode, CN 76k of 102k, JP 17.6k of 23.7k, KR 0.7k of 0.9k. Cause: grids kept "off" by the pulse / gate holds, and LNG short in the same weeks (SEA 62%, CN 44%, KR 88%). The plan also burns only 27k at SEA (78k nominal), so SEA is limited by gas, not crude.
- Waste instead of shortage: crude disposed at terminals per episode KR 5.0k, JP 4.8k, CN 3.3k, IN 24.5k = 37.6k of 215k shipped (17%). Plan ships 30k less and burns 9.5k more at KR+JP+CN: its extra crude is no extra supply, it is the absence of waste plus timing.
- Gulf source stock sits full (13.2k) from week 14: supply is not binding either.

## US edge (TW + SEA)
- Edge sea.tb.src_us_crude.chk_panama (cap 915): mean fill 17% in SEA's short weeks, never at 97%. v2 sends 9.7k to TW and 14.9k to SEA, the plan 25.2k in all (pooled at the strait), the same total as v2 (24.5k). TW gets 95% of its nominal burn while SEA gets 28%; SEA's crude sits at term_sea (about 4,500 all episode). The plan's -3.8k at TW / +4.8k at SEA is a split of an equal total, but SEA's problem is the valve, not edge capacity (the edge is 83% empty).

## 2. KR gas (lng)
- Not waiting at straits at the end (chk_malacca 1.4k + chk_taiwan 1.0k, all grids), not taken by others on the shared Qatar edge (edge 0: 65% full overall, 51% in KR's short weeks, 8 short weeks per episode, 11.9k deficit).
- In KR's short weeks the gas is on hand: opening terminal 3,159 + grid 1,147 against a cap of 2,750. term_kr disposes 28.2k lng per episode. v2 moves 263k terminal-to-grid against the plan's 267k and 138.6k on the Australian lane against 132.6k: the 5.5k difference is a 2% timing difference in the terminal-to-grid valve (rationing threshold), not routing.

## 3. Small (root 444, 40 episodes)
Shared first edges are 43-48% full: Qatar lng (TW, KR, EU) 48%, Gulf crude (KR, JP) 43%, US lng (TW, JP) 48%. In grid short weeks the edge is >=97% full in 2-8% (US lng edge: 29% for TW, 39% for JP). Same picture: no edge competition.

## Variants tried
None coded. Every ordering rule that could implement "yield to a short grid on the same edge" is vacuous here: a lower-priority grid already orders after the short ones, and its leftover edge capacity is not wanted by them (the downstream straits Malacca/Taiwan, with queues of 1-3.8k, or the valve bind them). The blunt storage cap was tried before (-0.0004 to -0.0084 on Small). No costs.py line exists.

## Check
Not run (no agent change).

## What to try next
1. The terminal-to-grid valve (FuelRules._hold / gate): in the short weeks of SEA, CN and JP 70-80% of the deficit is already at the terminal. Measure how many of those weeks are deliberate "hold"/"bank" weeks (trace=True) and whether the hold is wrong when the other fuel (lng) also cannot cover the week; SEA is lng-limited on both v2 and the plan.
2. Disposal of 37.6k crude at terminals (17% of what is shipped): ordering the position target against the terminal+grid storage, but only for Hormuz-only lanes (IN) and not as a cap on others.
3. Seven episodes are few; the figures above for the plan come from 7 episodes.

## Brief corrections
- The plan's crude is not additional supply: it ships 30k less than v2 on the Gulf edge.
- "The plan lands 3,800 less at TW": TW's gap and SEA's gap are one pool of 25k split differently; the edge is 83% empty.
- Source stock is at storage from week 14 (Gulf, US, gas): supply and first-edge capacity are not binding.

# Part 2: the terminal-to-grid valve and fuel thrown away at terminals (interim)

## Step 1: recoverable disposal (Full, root 111, 64 episodes, v2; tools/rec.py)
Disposed at the terminal, with room in the grid's own storage at that moment, and the same grid shedding for lack of that fuel in the next H weeks. Units per episode (1 unit = 4.13 M USD burned in a short week); upper bounds: the same deficit can be counted by several disposal weeks.

| grid fuel | disposed/ep | room at grid | same week | next 16 wk | next 32 | next 104 |
| --- | --- | --- | --- | --- | --- | --- |
| tw lng | 14,775 | 12,354 | 0 | 0 | 94 | 784 |
| kr lng | 9,275 | 8,624 | 0 | 183 | 817 | 1,187 |
| kr crude | 2,657 | 1,730 | 93 | 151 | 207 | 363 |
| jp lng | 21,169 | 20,834 | 0 | 737 | 1,203 | 2,702 |
| jp crude | 4,385 | 3,833 | 257 | 408 | 649 | 907 |
| cn lng | 53,670 | 52,431 | 0 | 41 | 302 | 1,110 |
| cn crude | 4,366 | 4,125 | 1,489 | 1,599 | 1,694 | 1,694 |
| us crude | 10,364 | 10,268 | 0 | 0 | 0 | 0 |
| eu lng | 97,824 | 95,219 | 0 | 0 | 0 | 0 |
| eu crude | 7,663 | 7,510 | 0 | 27 | 27 | 27 |
| sea lng | 62,355 | 62,073 | 0 | 1,053 | 3,094 | 6,086 |
| in crude | 30,644 | 10,329 | 3 | 17 | 33 | 107 |

Totals: 4,216 units (17 bn per episode, about 0.005 score) within 16 weeks; 14,968 units (62 bn) within 104 weeks. Almost all lng is disposed in weeks that are not near a shortage of it (EU, CN, SEA lng: 0-1,000 of 50-100k). The recoverable part is concentrated in CN crude (1.6k, same week), SEA lng, JP lng and JP crude. The mode per week (trace run) was NOT measured.

## Step 2: switches (both in FUEL, off = None)
- `hold_buffer` (share of terminal storage): in modes hold, bank and chold (fuel kept at the terminal for a complete stretch) a terminal fuller than this passes the surplus to the grid's own storage, as `grid_buffer` does in mode "on". One `elif` after the grid_buffer block in `FuelRules.fill`.
- `gate_keep` (weeks of burn): a banked or gated fuel keeps at the terminal only this many weeks of burn, the rest goes on to the grid.
- Not implemented: (iii) endgame release (end_weeks 16 already makes every grid "on" and the gate off in the last 16 weeks).

## Variants (costs.py, Full 32 episodes root 111, paired against v2)
| variant | score | difference | 90% interval | wins |
| --- | --- | --- | --- | --- |
| hold_buffer 0.8 | 0.8490 (base 0.8465) | +0.0025 | +0.0014 to +0.0038 | 21 of 32 |
| hold_buffer 0.6 | still running (log outputs/heur4/oil/run1.log) | | | |
| gate_keep 3.0 | queued in the same run | | | |
| gate_keep 1.5 | queued in the same run | | | |

Small x64 and `sbf check` were NOT run. The machine load average was 10; one Full x32 pair took about 10 minutes. The agent folder `outputs/heur4/agents/oil/` holds the code with `params.json` {"hold_buffer": 0.8} (best so far); `oil.diff` is the change to `fuel_part.py` only.

Update: hold_buffer 0.6, Full x32: 0.8437, difference -0.0029 (-0.0079 to +0.0012), wins 12 of 32. Not better than the base; a lower threshold passes too much; 0.8 is the best so far. gate_keep 3.0 is running (log outputs/heur4/oil/run1.log); gate_keep 1.5 queued.
Update: gate_keep 3.0, Full x32: 0.8512, difference +0.0046 (+0.0014 to +0.0090), wins 25 of 32. Best so far; agents/oil/params.json now {"gate_keep": 3.0}. gate_keep 1.5 running. Small x64, hold_buffer+gate_keep combined and sbf check not yet run.
Update: gate_keep 1.5, Full x32: 0.8494, difference +0.0028 (-0.0015 to +0.0077), wins 22 of 32; interval holds 0, so gate_keep 3.0 stays the best. Series finished.
