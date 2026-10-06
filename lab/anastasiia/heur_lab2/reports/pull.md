# pull: the chip chain as a pull system (heuristics lab 2), second version

Agent: `MAIN/outputs/heur2/agents/pull/` (`agent.py`, `fuel_part.py`, `chip_part.py`). Only `chip_part.py` differs from
`agents/anastasiia_rules_fuelchip` (the other two files are the base's, byte for byte); the unified diff is
`.claude/worktrees/agent-a8e9a684180d8feb6/handback/pull_chip_part.diff` in my worktree, and every variant folder, script and
array is in the same worktree. Switches: `PARAMS["pull"] = False` and `PARAMS["stock_fix"] = False` together give the base's
cost to the cent (checked on episodes 30, 7 and 3 of Small root 444 and episodes 0 and 7 of Full root 111).

This version adds to the first one the wafer-order fix the lead asked for on 6 October (a bug in the base's `_wafer_flows`,
see "The lead's note"). Everything is measured with the code in that folder unless a table says otherwise.

## Result in one place

| set | base | pull | paired difference, 90% interval |
| --- | --- | --- | --- |
| Small, root 111, 64 episodes | 0.7821 | 0.7861 | **+0.0040 (+0.0026 to +0.0054)** |
| by harm level 1 / 2 / 3 / 4 | 0.736 / 0.824 / 0.832 / 0.802 | 0.739 / 0.828 / 0.838 / 0.806 | |
| Full, root 111, 64 episodes | 0.7837 | 0.7861 | **+0.0023 (+0.0010 to +0.0039)** |
| by harm level 1 / 2 / 3 / 4 | 0.809 / 0.782 / 0.710 / 0.777 | 0.811 / 0.785 / 0.713 / 0.777 | |
| Full, root 111, 32 episodes | 0.7887 | 0.7899 | **+0.0012 (+0.0001 to +0.0025)** |
| Full, root 111, 16 episodes | 0.7705 | 0.7717 | **+0.0012 (-0.0004 to +0.0036)** |
| Small, root 444, 40 episodes, USD bn per episode (diagnostic) | 2,734.9 | 2,732.4 | -2.5 (about +0.0023) |

`sbf check` passes on Small, Full and Tiny (no docker run): longest week 0.011 s on Small (budget 2 s), 0.043 s on Full
(budget 4 s); Tiny 0.004 s. Two plays of the same episode give the same cost to the cent (Small episode 30 of root 444 twice;
Full episodes 0 and 7 of root 111 in two separate processes); `ruff check` and `ruff format --check` pass.

What the rules did to the waste on the two boards (`account.py`, arrays in `outputs/pull/`; Small root 444 episodes 0..39,
Full root 111 episodes 0..7):

| per episode | Small, base -> pull | Full, base -> pull |
| --- | --- | --- |
| wafers disposed at fabs, M | 0.43 -> 0.14 | 8.96 -> 1.46 |
| raw chips disposed at fabs, M | 2.66 -> 2.60 (chip_le_raw 1.79 -> 1.77, chip_mat_raw 0.87 -> 0.83) | 11.03 -> 10.71 (chip_le_raw 5.78 -> 5.78, chip_mat_raw 5.24 -> 4.93) |
| disposal cost, USD bn | 4.4 -> 4.2 | 20.5 -> 18.1 |
| energy to fabs, GWh | 13,813 -> 13,792 | 82,748 -> 82,244 |
| lots started, M | 10.31 -> 10.25 | 67.36 -> 66.71 |
| chip_le / chip_mat sold, M | 12.73 / 4.69 -> 12.79 / 4.63 | 33.35 / 39.29 -> 33.33 / 39.04 |
| total J, USD bn | 2,734.9 -> 2,732.4 | 7,966.1 -> 7,967.3 (8 episodes, single episodes move by +-10 bn; the 16 and 32 episodes give +0.0012 of RSS) |

The gain is small and has two sources. On Small it sits in the episodes in which a wafer source's supply is cut: the wafers are
kept for the fabs whose outlets need them (24 of 40 episodes of root 444 are better, 9 worse). On Full the wafer-order fix
removes 84% of the wafers the base threw away (8.96 -> 1.46 M an episode, 2.4 bn of disposal) and the pull rules add a little;
the raw chips thrown away at the fabs hardly change (-3%): that is a route and storage limit, not a wafer-order bug (see "The
lead's note"). The folders are in my worktree: `agents/pull_fin` (the final agent, installed as `pull`), `agents/fin_off` (the
same with `pull` and `stock_fix` off), `agents/g_rc1` and the other variants of the tables below, each with its `params.json`;
the scripts and the arrays are in `lab/pull/` and `outputs/pull/`.

## What I changed (all in `chip_part.py`)

### 1. The wafer order must use the stock there is (`stock_fix`; the lead's third fact)

`_wafer_flows` restored a fab's wafer stock to its target as if the fab started `min(cap_eff, wafers)` lots this week. At a fab
whose grid is dark nothing starts, the same subtraction repeats the next week, and a fab at its storage limit was sent about a
fifth of its storage every week; the simulator throws the excess away (Full: 8.96 M wafers an episode, cn_mature 4.27 M and
sea_mature 2.08 M of them). Now the order is also cut to `room = storage - (stock + wafers arriving by next week - starts)`,
where the starts are counted only if the fab's grid served its base load last week (`last_week.shed.qty`). So the order uses the
stock actually there and never exceeds the wafer storage on the week it looks at (what is still thrown away, 1.46 M an episode on
Full, is sea_mature 0.61 M, kr_memory 0.34 M, tw_mature_2 0.12 M, jp_memory 0.10 M and the rest below 0.1 M each; I think it is
shipments with a lead above one week that land on a full store, because the order counts only arrivals by next week; sea_mature's
grid is dark in 91% of the weeks). Tried and rejected: counting every shipment on its way against the storage (`pull_f2`: Full
-0.0002 and the level-4 episodes score 0.011 lower, though fewer wafers are thrown away), and assuming no start in a dark fab for
the target as well (0.52 M wafers thrown away instead of 1.46 M, but 66.34 M lots started an episode instead of 66.71 M: the
buffer a dark fab keeps for its next window is smaller; I dropped it for the lots and did not score it alone).

### 2. Wafers of a source whose supply is cut are kept for the fabs whose outlets need them (`pull`)

A wafer source whose supply is cut holds all the wafers there will be: at week 1 in 30 of the 40 episodes of root 444 and 50
of the 64 of root 111 on Small (JP wafer 12/40, UA neon 14/40, DE wafer 13/40 on root 444), 12 of 16 on Full; the README
and the base agent do not know it. The base asks every fab for three weeks of capacity in week 1, so the stock is drained at
once, pro rata to what every fab asks (the cheap chip's fabs take their share), and the lots start while the chain is still full
of the initial work in process (9.05 M chip_le in the chain at reset in episode 30, wafers at fabs and market stocks counted).
Later the fabs that feed the idle outlets have no wafers (episode 36: osat_kr's outlet of 142,000 a week idle from week 20 to 47;
the plan starts KR lots in weeks 26 to 37). The rules (`PARAMS["pull"]`):

1. a wafer request never exceeds what its source holds; fabs are served the dearer chip first, within a chip those whose grid
   served its base load last week first (`power_order`); routes from cut sources come last in a fab's list;
2. `wafer_sources`: a source is cut when `graph_now.supply.avail` is below `scarce_frac` (0.5) of its nominal `supply_rate` (read
   from `config`, like every other number);
3. `reserve_dear`, `reserve_useful`: the stock of a cut source goes only to the fabs of the dearest chip that can still reach a
   market; the cheaper chip's fabs take nothing from it;
4. `pull_plan`: what a fab may draw from a cut source is capped by what it must still make. Two max-flows per chip (dearer chip
   first, the cheaper on what is left of shared edges), each over a window of weeks with every capacity a week's capacity times
   the window. The first window is as long as a wafer needs to become a sold chip and sells what the chain already holds
   (wafers and work in process at fabs, raw chips at fabs, on the way to plants and at plants, work in process and packaged chips
   at plants, chips on the way to markets and at markets) through the routes that are alive. The second is `pull_buffer` (12)
   weeks, sells what is left, and then adds production fab by fab (fewest plants first, then lowest energy per lot) where an
   outlet is still idle; that is what each fab must still make. So wafers wait at the source until the outlets they would feed
   are about to run dry.

### 3. A cap on the weekly wafer order (`rate_cap`, built, tested, left off)

A fab can be sent at most `rate_cap` times its sale rate in wafers a week (`sale_rates`, the chips a fab can still sell). It is
the only rule that touches the raw chips thrown away at fabs on Full, and it is a knife edge: 1.0 gives Full +0.0031 (+0.0016 to
+0.0046) on 32 episodes and +0.0036 (+0.0022 to +0.0050) on 64 (the final agent: +0.0012 and +0.0023) and Small +0.0021
(-0.0007 to +0.0046; without the cap Small is +0.0044), 0.8 loses 0.011 on Full, 0.6 loses 0.07.
With the cap at 1.0 (8 Full episodes, against the final agent) the raw chips thrown away fall from 10.71 to 5.75 M, the fabs'
energy by 7.1 TWh and the disposal cost from 18.1 to 11.3 bn, but 0.17 M fewer chip_le are sold and the shortage rises by 6.2 bn. Exempting the fabs whose
grid is dark from the cap changes nothing (Full +0.0031 at 1.0, -0.0101 at 0.8). I left it off: on Full it adds +0.0013 to the
final agent's +0.0023 (64 episodes, the second half of them new to it, so the gain is not a fit to the first 32), but the 1.0 was
picked among five settings on one root (the interval does not count that choice), the cliff is just below it (0.8 loses 0.011), and
Small loses 0.0023 of what the final agent gets there. It stays in `PARAMS["rate_cap"]` (default 0), and `{"rate_cap": 1.0}` in a
`params.json` beside `chip_part.py` switches it on; if the private board is all that counts and a second root agrees, that is the
variant to take.

Functions touched: `pull_plan`, `wafer_sources`, `_lead_to_sale` (new), `_wafer_flows`, `__init__` (supply tables, `grid_ix`,
`fab_inputs`), `_context` (`avail`, `shed`), `PARAMS`. `usefulness`, `sale_rates`, `chip_value`, `_pack_flows`, `_raw_flows`
and the fuel part are untouched.

## The lead's note of 6 October

1. "On Full the power shed already equals the clairvoyant's; the whole remaining gap is chips." Consistent with what I see: the
   chips lost sit in the outlets and the routes, not in the lots (chip_le's alive outlet capacity of the plants is 377,000 a
   week of 715,000 nominal and the agent ships 315,000 of it; the five fabs of the US grid start whatever wafers they hold, the
   grid serves its base load in 93% of the weeks). What the chip chain gave on Full, measured with `compare.py --task=full`: +0.0023
   (64 episodes) for the final agent and +0.0036 with the rate cap at 1.0 (see the tables). +0.0023 is about 8 bn an episode,
   against the 855 bn between the base and the clairvoyant on the eight episodes of the account (7,966 and 7,111 bn): these rules
   do not reach most of that gap.
2. "Fabs dispose of 5.8 M chip_le_raw and 5.2 M chip_mat_raw an episode." Confirmed (11.03 M, 20.5 bn of disposal and about 14
   TWh of fab energy in disposed chips). It is not caused by the wafer order: the disposal is at the five small leading-edge fabs
   of the US, EU and ROW (us_leading_1..3, eu_leading, row_leading: each throws away 0.4 to 0.6 M raw chips, half of what it
   starts; raw storage 54,000 to 88,000 chips; the routes to the plants are cut to 25%), and in a few episodes at the big ones
   (kr_memory 8.5 M and 6.6 M in episodes 2 and 4, eu_mature 7.2 M and tw_mature_2 5.5 M in episode 7). A fab can only be
   stopped through its wafer order, and every cap on it that I tried (idea 1, 2, `rate_cap`) loses chips at the same time.
3. "Wafers are disposed of at dark fabs." Confirmed and fixed: 8.96 -> 1.46 M an episode (see `stock_fix`); cn_mature 4.27 -> 0.04 M,
   sea_mature 2.08 -> 0.61 M, jp_memory 0.64 -> 0.10 M, kr_memory 0.63 -> 0.34 M.

What the whole disposal is worth on Full: 20.5 bn of disposal cost plus 14 TWh of fab energy (7 to 11 bn at the 0.5 to 0.8 bn a
TWh I measured) is about 30 bn an episode. Naive minus clairvoyant averages 3,520 bn over these eight episodes (the cached
references), so even a rule that removed all of it at no cost in sales would add at most about +0.009 of RSS. The rule that
removed half of it (rate cap 1.0) added +0.003.

## Every variant tried

Small, root 111, 32 episodes (base 0.7843), `compare.py`, unless said otherwise; variants 1 and 2 were overwritten and are
described here, the others are folders in my worktree.

| # | variant | score | vs base | 90% interval | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | order-up-to on the inventory position of each chip (everything downstream of the wafer source), orders split by a lexicographic joint max-flow, buffer 5 weeks | 0.7305 | -0.0538 | -0.0704 to -0.0403 | much worse |
| 2 | the same position, outlet-aware (chips per plant counted up to what its outlets ship in the horizon), used as a brake on the usual order, buffer 12, ramp 3 weeks | 0.7780 | -0.0062 | -0.0100 to -0.0032 | worse |
| 3 | diagnostic, six episodes of root 444, no `compare`: no wafer orders before week 12 | | dJ against base, bn: -0.6, +128.5, +17.4, +19.8, +49.4, +86.4 (episodes 30, 3, 33, 22, 4, 26) | | not a rule |
| 4 | `pull_plan` on cut sources, `reserve_dear`, source-aware asks, buffer 6 | 0.7872 | +0.0030 | +0.0005 to +0.0055 | helps |
| 5 | same, buffer 3 | 0.7864 | +0.0021 | -0.0005 to +0.0047 | |
| 6 | same, buffer 12 | 0.7885 | +0.0042 | +0.0024 to +0.0062 | best |
| 7 | same, buffer 24 | 0.7884 | +0.0041 | +0.0024 to +0.0060 | same as 12 |
| 8 | buffer 6, `scarce_frac` 0.8 | 0.7872 | +0.0030 | +0.0005 to +0.0055 | same as 0.5 |
| 9 | buffer 12 and the plan's cap on every source (`pull_all`) | 0.7804 | -0.0038 | -0.0088 to +0.0011 | worse |
| 10 | #6 + `reserve_useful` + `power_order`, paired with #6 | 0.7887 | +0.0002 | +0.0000 to +0.0005 | kept (neutral, principled) |
| 11 | #10 = the first version of this agent, 64 episodes (base 0.7821) | 0.7861 | +0.0040 | +0.0027 to +0.0054 | |
| 12 | #10 on Full, 16 episodes (base 0.7705) | 0.7705 | -0.0000 | -0.0012 to +0.0016 | no effect on Full |
| 13 | #10 + `stock_fix` (**final**), 32 episodes (base 0.7843) | 0.7887 | +0.0044 | +0.0027 to +0.0064 | the agent |
| 14 | #13 + `rate_cap` 1.0 | 0.7863 | +0.0021 | -0.0007 to +0.0046 | worse than 13 on Small |
| 15 | #13 on 64 episodes (base 0.7821) | 0.7861 | +0.0040 | +0.0026 to +0.0054 | the agent |

Full, root 111, 32 episodes (base 0.7887), `compare.py --task=full`:

| variant | score | vs base | 90% interval |
| --- | --- | --- | --- |
| pull rules alone (the first version of this agent) | 0.7892 | +0.0005 | -0.0007 to +0.0018 |
| `stock_fix` alone (`pull` off) | 0.7894 | +0.0007 | +0.0003 to +0.0011 |
| both (**final**: storage counted with the starts there will be, arrivals by next week; `pull_g`, the final code) | 0.7899 | +0.0012 | +0.0001 to +0.0025 |
| the final agent itself (`pull_fin`, the same code with a longer docstring), 16 episodes (base 0.7705) | 0.7717 | +0.0012 | -0.0004 to +0.0036 |
| the final agent, 64 episodes (base 0.7837; the 32 above are its first half) | 0.7861 | +0.0023 | +0.0010 to +0.0039 |
| the final agent and `rate_cap` 1.0 (`g_rc1`), 64 episodes (base 0.7837) | 0.7873 | +0.0036 | +0.0022 to +0.0050 |
| both, every shipment on its way counted against the storage (`pull_f2`) | 0.7885 | -0.0002 | -0.0018 to +0.0015 |
| `pull_f2` and `rate_cap` 1.0 | 0.7899 | +0.0012 | -0.0004 to +0.0028 |
| `pull_f2` and `rate_cap` 1.5 | 0.7883 | -0.0004 | -0.0027 to +0.0017 |
| final and `rate_cap` 1.0 | 0.7918 | +0.0031 | +0.0016 to +0.0046 |
| final and `rate_cap` 1.0, fabs without power not capped | 0.7918 | +0.0031 | +0.0017 to +0.0046 |
| final and `rate_cap` 0.8 | 0.7774 | -0.0113 | -0.0184 to -0.0053 |
| final and `rate_cap` 0.8, fabs without power not capped | 0.7786 | -0.0101 | -0.0169 to -0.0043 |
| final and `rate_cap` 0.6 | 0.7163 | -0.0724 | -0.0858 to -0.0596 |
| final and `rate_cap` 1.0 and the plan's cap on every source (`pull_all`) | 0.7656 | -0.0231 | -0.0280 to -0.0181 |

The four ideas of the brief: 1 (joint sellable rate) and 2 (order-up-to on the chain's position) are variants 1 and 2, then
the plan's max-flows; 3 and 4 were checked on the data and not built:

- Idea 3 (raw routing): of the 2.66 M raw chips disposed at fabs per episode on Small, 1.41 M were disposed in weeks when some
  reachable plant still had packaged-stock room at the end of the week. That is an upper bound: the room is the same room every
  week and refills only at the plant's outlet rate, and `_plant_state` already counts the outlets' drain. Episode 33 (KR holds
  1.1 M raw chips while osat_my's outlet idles) is a route limit, not the room rule: the only open route KR to osat_my carries
  160,123 a week. On Full the feeders that hold chips have saturated routes (tw_leading 40,000 of 40,000, us_leading 1,000 of
  1,000 and 6,000 of 6,000) and the others hold none: osat_my is short of about 15,000 a week that cannot be moved.
- Idea 4 (`_pack_flows`): the outlet capacity left unused in weeks when the plant held packaged chips is 165,000 chip_le and
  739,000 chip_mat per episode on Small (base agent, root 444, 40 episodes), and that counts markets whose stock already covers
  their need; on Full, with chip_le's use of the shared air edges taken off, chip_mat's idle outlet capacity while a plant holds
  chips is 11,600 a week in all (3% of its shipments) and chip_le's 4,500 (1.4%): nothing to take there either.

## `account.py`, root 444, episodes 0..39, `--plan` (base: `reports/base_444.txt`)

| per episode | base | pull (final) |
| --- | --- | --- |
| total J, USD bn | 2,734.9 | 2,732.4 |
| shortage / disposal / shed, USD bn | 905.2 / 4.4 / 1,810.6 | 902.8 / 4.2 / 1,810.6 |
| chip_le sold, M (all markets) | 12.73 | 12.79 |
| chip_mat sold, M | 4.69 | 4.63 |
| lots started, M | 10.31 | 10.25 |
| wafers disposed at fabs, M | 0.43 | 0.14 |
| raw chips disposed at fabs, M | 2.66 (chip_le_raw 1.79, chip_mat_raw 0.87) | 2.60 (1.77, 0.83) |
| energy to fabs, GWh | 13,813 | 13,792 |
| free segment lost, GWh | 6,193 | 6,210 |
| gap to the best plan, USD bn | 141.6 = chips 66.3 + shed 68.0 + other 7.3 | 139.0 = chips 63.9 + shed 68.0 + other 7.2 |
| "no wafers" share of capacity, weeks 1..38, TW lead / TW mat / KR / JP / EU lead / EU mat | 0.373 / 0.519 / 0.323 / 0.317 / 0.332 / 0.340 | 0.364 / 0.531 / 0.320 / 0.284 / 0.285 / 0.304 |

Episodes that moved (root 444, USD bn against the base): 13 -15.2, 36 -14.9, 30 -10.7, 26 -9.4, 20 -9.2, 1 -9.0, 0 -6.5, 7 -6.5,
29 -6.3, 15 -5.1 ... and 24 +3.7, 27 +3.4, 2 +2.3, 35 +1.5; better in 24 of 40 episodes, worse in 9. All the large gains are
episodes with a cut source; the cut sources are used for the chips whose outlets need them, not in week 1. The lots, the energy
and the disposal of the fabs did not move: the rule re-times and re-allocates the wafers of cut sources, it does not start fewer
lots. The wafer-order fix is worth 0.3 M wafers an episode on Small (0.43 -> 0.14 M, 0.1 bn): nothing there.

## What the data says (root 444, the base agent against the best plan)

- The agent wastes about 3.0 M more chip_le per episode than the plan: 1.89 M disposed (all chip_le slots) and 1.13 M more left
  in the chain at the end (632,000 raw at fabs and 725,000 packaged at plants, against 12,000 and 219,000). Part of it cannot be
  reached: the plan holds raw chips at plants, but the simulator packages min(throughput, raw) at once and the packaged stock
  above storage is disposed of. In episode 30 about 1.9 M of the 2.08 M sales gap is this.
- In outlet-limited episodes (2, 4, 5, 11, 12, 21, 22, 25, 32, 37) the agent sells what the plan sells with about twice the lots,
  but its flooding is what keeps every plant stocked, so every outlet works in all 52 weeks (episode 22: the three plants'
  packaged stock stays at storage and the outlets of 73 + 96 + 67 thousand a week are all used). The plan feeds each outlet just
  in time from the fab that can reach it (TW leading and TW mature for osat_tw); a rule that holds the lots back on the chain's
  total position cannot do that, and the two I built lost 1.7 to 3 M chips in the episodes where they bound (variant 1: -0.054;
  variant 2: -0.006). The wafer-order cap of `rate_cap` shows the same on Full.
- The energy a lot saves is worth much less than the README says. Variant 2 cut the fabs' energy from 13.8 to 11.7 TWh and the
  shed fell by 1.7 bn, not by 8.6 bn (2.1 TWh at 4.1 bn a TWh): the grids burn less fuel instead of saving it (free segment lost
  6.2 to 7.4 TWh). On Full `rate_cap` 1.0 took 7.1 TWh off the fabs and the shed fell by 3.3 bn (0.5 bn a TWh). The saving is real
  only if the fuel part turns the free segment into saved gas.
- Raw chips disposed at fabs on Small: about half is initial work in process maturing in weeks 1 to 9, when plants and outlets are
  full (episode 30, KR: 905,000 in weeks 5 to 9); the rest are lots started into a saturated chain.

## What in the README turned out wrong or incomplete

1. "Every such lot burns grid energy worth 5,000 to 8,000 USD of base load": only through the fuel part; measured about 0.8 bn a
   TWh with the present fuel part on Small and 0.5 on Full, not 4.1.
2. "Start nothing in the outlet-limited episodes" (the plan starts 0.11 M lots in episode 11): the same chips are sold, but the
   plan has foresight and relaxations (it keeps raw chips at plants, which the simulator does not allow); a rule on the chain's
   inventory position lost sales (variants 1, 2, 9), and so does a cap on the weekly order (`rate_cap` below 1.0).
3. The README does not mention that wafer sources are cut at week 1 in three episodes of four (Small) and in 12 of 16 (Full):
   that is where the chip chain still had money on Small.
4. "About 6 M chip_le are already in the chain at reset": 9.05 M in episode 30 if the wafers at fabs and the market stocks count.
5. "The agent holds no raw chips at plants (the plan: 119,000)": see 2; the plan's raw stock at a plant becomes packaged stock in
   the simulator the same week.
6. The raw chips thrown away on Full (fact 2 of the lead's note) are not caused by the wafer-order bug (fact 3): they are a route
   and storage limit of the small fabs, and the wafer order can only reduce them by starting fewer lots, which also sells fewer
   chips.

## What I would try next

1. A reactive cap only where the waste is, instead of a cap on every fab. `rate_cap` 1.0 shows that about 5 M raw chips an
   episode can go at the price of 0.17 M chip_le sold less (8 Full episodes, `account.py`, fab by fab): the saving sits at
   kr_memory 1.2 M, eu_mature 1.1 M, the five small leading-edge fabs 1.1 M together, us_mature_1 0.9 M and tw_leading 0.5 M. Only
   the five small fabs are limited by their own routes (a cap at the capacity of a fab's raw routes would catch just those 1.1 M);
   at the others the plants do not take the chips. So the cap should follow what the chain takes from a fab: wafers ordered at
   most at a moving average of the raw chips the fab shipped in the last weeks, plus its free raw storage spread over the lead
   time. I expect (not measured) that it keeps most of the saving (7 bn of disposal, 7 TWh) and loses less of the 6 bn of sales.
2. A time-expanded flow (weekly, not a window total) in `pull_plan`: a total cannot tell an outlet that runs dry in week 20 from
   one that runs dry in week 10. With it the brake of variants 2 and 9 could be applied to outlet-limited episodes without losing
   sales, and the energy saved would count once the fuel part keeps the free segment.
3. Let chip_mat have what chip_le's long-window need leaves of a cut source (it gets nothing now: chip_mat sold -0.06 M an
   episode, -0.4 M in episode 26).
4. The agent ends an episode with 632,000 raw chips at fabs and 725,000 packaged at plants: lots that mature in the last 15
   weeks are worth only their energy; the plan's window could be cut at the horizon to stop them.
5. Merging with `throttle` (fuel part): the files are disjoint, but the throttle changes when a grid is dark, and `stock_fix` and
   `power_order` read `last_week.shed.qty`; run `compare.py` on the merged folder before trusting the sum.

## For `hub/tried/heuristics.md` and `hub/FINDINGS.md` (ready to paste, in Ukrainian; not written there)

Для `hub/tried/heuristics.md`, новий розділ «Чипи: вафлі як дефіцитний ресурс (`pull`, лаб heur2)»; таблиця «хто | що | як міряли | результат | вердикт»:

| хто | що | як міряли | результат | вердикт |
| --- | --- | --- | --- | --- |
| anastasiia (сесія `pull`) | вафлі з джерела, чиє постачання обрізане (менше половини номіналу), ідуть лише фабрикам дорожчого чипа, які ще мають що продати: два max-flow на чип (вікно «від вафлі до продажу» і ще 12 тижнів), запит не більший за запас джерела; `PARAMS["pull"]` | Small 111 ×64 проти `anastasiia_rules_fuelchip`; Full 111 ×32 | Small +0.0040 (+0.0027…+0.0054); Full +0.0005 (−0.0007…+0.0018) | допомогло на Small, усі рівні шкоди кращі |
| anastasiia (сесія `pull`) | `stock_fix`: замовлення вафель вміщується в склад; фабрика, чия мережа минулого тижня не покрила базове навантаження, нічого не стартує (база віднімала `started` завжди і слала «мертвій» фабриці п'яту частину складу щотижня) | Full 111 ×32; викинуті вафлі — Full 111 ×8, разом із `pull` | +0.0007 (+0.0003…+0.0011); вафель викинуто 8.96 → 1.46 млн за епізод | допомогло, але мало |
| anastasiia (сесія `pull`) | обидва разом, код `outputs/heur2/agents/pull` (лаб-копія; до `agents/` лише після Formal Results), змінено лише `chip_part.py` | Small 111 ×64; Full 111 ×64, ×32, ×16 | Small +0.0040 (+0.0026…+0.0054); Full ×64 +0.0023 (+0.0010…+0.0039), ×32 +0.0012 (+0.0001…+0.0025), ×16 +0.0012 (−0.0004…+0.0036) | кандидат; паливо не чіпає |
| anastasiia (сесія `pull`) | порядок до рівня на позиції запасу ланцюга (усе нижче за джерелом вафель) | Small 111 ×32 | −0.054 (−0.070…−0.040); як гальмо з урахуванням виходів −0.006 (−0.010…−0.003) | зашкодило: заливання тримає виходи зайнятими щотижня |
| anastasiia (сесія `pull`) | обмеження плану на вафлі з усіх джерел (`pull_all`) | Small ×32; Full ×32 разом зі стелею 1.0 | −0.0038 (−0.0088…+0.0011); −0.0231 (−0.0280…−0.0181) | зашкодило |
| anastasiia (сесія `pull`) | не замовляти вафлі до тижня 12 | діагностика, 6 епізодів root 444 | +300 млрд USD сумарно проти бази | ні |
| anastasiia (сесія `pull`) | стеля тижневого замовлення вафель: κ × темп продажу фабрики (`rate_cap`) | Full 111 ×32; Small 111 ×32 | Full: κ=1.0 +0.0031 (+0.0016…+0.0046) на 32 і +0.0036 (+0.0022…+0.0050) на 64 (без стелі +0.0023), κ=0.8 −0.0113, κ=0.6 −0.0724; Small κ=1.0 +0.0021 (без стелі +0.0044) | край прірви: вимкнено; вдвічі менше викинутих сирих чипів, але менше продажів |

Для `hub/FINDINGS.md`, розділ «Де губляться чипи й паливо» (джерела: `account.py`, `compare.py`, `outputs/heur2/reports/pull.md`; перевірено однією людиною):

- **Джерела вафель обрізані на тижні 1** (постачання менше половини номіналу, `graph_now.supply.avail`) у 30 з 40 епізодів root 444 і 50 з 64 root 111 на Small, у 12 з 16 на Full. Запас у джерелі тоді — усе, що буде. База забирає його в тижні 1 пропорційно запитам усіх фабрик і запускає партії, поки ланцюг повний початкового WIP (9.05 млн chip_le у ланцюзі при reset, епізод 30 root 444, разом із вафлями на фабриках і запасами ринків).
- **Помилка замовлення вафель у базі.** `started = min(cap_eff, на складі + прибуття)` віднімалася й тоді, коли фабрика без живлення нічого не стартує, тож фабрика на межі складу отримувала п'яту частину складу щотижня, і симулятор її викидав. Full 111 ×8: 8.96 млн вафель за епізод (cn_mature 4.27, sea_mature 2.08, jp і kr memory по 0.6), після `stock_fix` і `pull` 1.46 млн (sea_mature 0.61, kr_memory 0.34). Small 444 ×40: 0.43 → 0.14 млн.
- **Викинуті сирі чипи на Full (11.0 млн за епізод: chip_le_raw 5.8, chip_mat_raw 5.2; 20.5 млрд USD утилізації і 14 ТВт·год енергії фабрик, тобто 7–11 млрд USD) — не помилка замовлення вафель,** а межа маршрутів і складу: п'ять малих фабрик (us_leading_1..3, eu_leading, row_leading) викидають по 0.4–0.6 млн, половину випуску (маршрути до заводів урізано до 25 %, сирий склад 54–88 тис.), і кілька епізодів великих (kr_memory 8.5 і 6.6 млн в епізодах 2 і 4, eu_mature 7.2 і tw_mature_2 5.5 млн в епізоді 7). Усе це разом ≈ 30 млрд USD за епізод, тобто 0.009 від різниці naive − clairvoyant (3 520 млрд, Full 111, епізоди 0..7): це стеля виграшу від правила, що прибрало б усе без втрат у продажах.
- **Стеля замовлення** `rate_cap` 1.0 вдвічі зменшує викид (10.7 → 5.8 млн) і бере 7.1 ТВт·год з енергії фабрик, але shed падає лише на 3.3 млрд USD (≈ 0.5 млрд за ТВт·год, не 4.1 з README), а продається на 0.17 млн chip_le менше (shortage +6.2 млрд). Нижче 1.0 штраф різко зростає: 0.8 дає −0.011, 0.6 дає −0.07 на Full.
- **Енергія лота коштує мало:** ≈ 0.8 млрд USD за ТВт·год на Small (гальмо на позиції запасу: −2.1 ТВт·год → shed −1.7 млрд) і ≈ 0.5 на Full. 4.1 млрд з README — лише якщо fuel part повністю перетворить зекономлену енергію в збережений газ.
- **Full, виходи заводів:** жива потужність виходів для chip_le 377 тис. за тиждень проти номінальних 715 тис., агент везе 315 тис. (84 %). Невикористана потужність виходів, поки завод має пакетовані чипи (з урахуванням спільних повітряних ребер), — 11.6 тис. за тиждень для chip_mat (3 % відправок) і 4.5 тис. для chip_le (1.4 %): `_pack_flows` грошей не лишає.
