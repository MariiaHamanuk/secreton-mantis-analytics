# Mechanics lens: what the simulator's own rules still leave on the table

10 October. Read-only audit of `agents/anastasiia_plan_hull3/sbfv/dynamics/` (sim, clip, chokepoint, production,
cost, state), `plan_core.py` (`actions`, `cell`, `_lots`, `cost`) and `agents/anastasiia_plan_hazard*/agent.py`
(`act`, `_wishes`, `_ask_early`, `_ask_scaled`), plus accounting on the kept records
`outputs/hazard_lab/play/w50as_{s,f}_444.pkl` (the second candidate without a clock; Small 444 x64, Full 444 x16).

**Status of every number below:** accounting on kept records, one process, one pass, not checked a second time. None
is a paired score. bn = 1e9 USD an episode. Rough conversion from FINDINGS: 0.01 of score is about 11 bn on Small and
34 bn on Full (the levels' weights make this approximate). Each measure states whether it is an upper bound.

Scripts (temporary, outside the repo): `<scratchpad>/an1.py … an14.py`, `dump_inst.py`, `dump_edges.py`, where
`<scratchpad>` is a scratch directory outside the repository; of its scripts only the accounting of idea 1 is kept, as `../lot_pool.py`.
They rebuild each scenario's marks (`sample_omega`, `compute_marks`, 1.2 s an episode) and read the pickles; the
container side of the lot book is replayed exactly from the kept dispatches with the stocks forced to the record
(queue totals match the record to 1e-10 on Full, 1.8 units on Small).

## 0. The short version

The audit found no large free lunch of the over-ask kind. The program inside the regimes already models nearly every
rule, and five plausible exploits measured as nothing (section 3). What is left:

| # | idea | kind | Small | Full |
| --- | --- | --- | --- | --- |
| 1 | a lot is a cheap option on an exit: whole weeks asked where exits have room | certain accounting, uncertain conversion | pool 20 bn "as observed" (48 with the future's exits) | 32 bn, two episodes of 16 carry 83 % |
| 2 | wafers steer the pro-rata split of a partial top between the fabs of one grid | certain mechanics, uncertain size | +1.9 bn | +4.8 bn |
| 3 | a margin of gas at the grid while its output is cut (the week an energy shock ends) | certain mechanics, zero downside | +1.0 bn | +3.4 bn |
| 4 | the uniform over-ask for tanker releases (strait throughput, cut out-edges) | certain mechanics, zero downside, not measured | est. 0.3–0.5 bn | est. 2–3 bn |
| 5 | gas throttled beside idle free generation, as a free hedge | certain mechanics, small | +0.7 bn | +0.7 bn |
| 6 | containers held upstream of a throughput-bound strait (FIFO, pro rata by quantity) | speculative | exposure only | exposure only |

## 1. Facts of the instances that close whole families

From `dump_inst.py` (both networks, the generator's instances):

- **Every market is a lost-sales sink** (`backlog False` on all 8 demands of Small and all 16 of Full): no backlog
  compounding to exploit. Penalties 50.40–50.52 thousand USD (`chip_le`), 10.28–10.40 (`chip_mat`).
- **Every grid is `base_first`** (4 on Small, 8 on Full): no proportional or industrial-first grid.
- Nuclear fuel never passes a strait (no `nucfuel` slot at any chokepoint): overrides, which go first in the tanker
  pool, displace nothing. Tanker and container cargo never share an edge (one pool an edge) nor a throughput.
- Every tanker term of the fleet pool has lane None, so the lane tag an override slot carries
  (`plan_core.py:351-353` takes the first slot of each (strait, cargo, edge)) changes nothing.
- Edges shared by two commodities among the action slots: only the valves (gas and crude) and the air exits of plants
  (`chip_le` and `chip_mat`).
- Stores: fab input and output 3.6 weeks of capacity; terminals and grids about 4 weeks of gas burn each; markets 4
  weeks of demand; **a strait has no cap and no disposal** (`sim.py:92`). Holding is the same at a terminal and at
  its grid (33.4 USD a GWh-week) and the same at a plant and at a market (16.2): the program is indifferent where
  such stock sits.
- Gas burn at the cap against the ration's threshold: TW 2 177 / 2 053, KR 2 750 / 2 357, JP 5 400 / 6 480,
  EU 8 840 / 10 608 GWh. Valves carry 2.8 weeks of burn a week on TW and far more elsewhere.
- Cost of an episode of `w50as` on Small: shed 1 750.5, shortage 799.6, tariffs 7.4, holding 6.7 (mostly the nuclear
  stock), disposal 1.8, freight 1.1, queues 0.5, war risk 0.05. Everything but the first two is 0.7 %.

## 2. Ideas, ranked by expected value

### 1. A lot is a cheap option on an exit: ask whole weeks where exits have room

**Kind:** certain accounting, uncertain conversion; planner-side, the largest number found.

1. **Rule.** Lots are `min(capacity, wafers on hand)` scaled by what is left after the base load
   (`sim.py:322-325`, `349-355`; `production.py:25-27`). A lot started in a week that would otherwise shed costs only
   its energy at the price of shed load: 5.2 thousand USD (memory), 8.3 (leading), 3.7 (mature) against a sale of
   50.4 or 10.3. An unsold chip costs 2 thousand USD of disposal, or nothing in a strait's queue.
2. **Mechanism.** Chips sold. For every `chip_le` lot short of capacity in a week without energy to spare (weeks 1 to
   T - 14) the script asks in turn: wafers on hand or within reach (room on an open direct edge and stock left at
   the source the week before); energy from bunching fuel the grid had **already burned by that week** (burn of its
   scarcest gate fuel in weeks of its cap, minus the top-weeks the fabs got); room on a direct edge to a plant the
   week the lot is out; an exit of that plant with room toward a market with unmet demand within 12 weeks of
   packaging. Every capacity is used once. Value is net of the top's energy at the price of shed load.

   | per episode | Small | Full |
   | --- | --- | --- |
   | lots short of capacity, weeks without energy to spare | 13.4 mn | 21.5 mn |
   | with wafers at hand or within reach | 10.5 mn | 21.0 mn |
   | and bunched fuel, a way to a plant, an exit with room later | 1.15 mn = 51 bn (median 37; 51 of 63 over 5 bn) | 1.33 mn = 59 bn (median 1.9) |
   | and that exit had the room in the week of the lot too | 0.46 mn = **20.4 bn** (median 6.5; 33 of 64 over 5 bn) | 0.72 mn = **32.2 bn** (episodes 9 and 2 carry 273 and 156) |

   The third row takes the episode's bunched fuel as one sum; with the order of weeks imposed, as in the fourth
   row, it is 48 bn on Small (`an14.py`) and was not recomputed on Full.

   Beside it, the fuel side alone: burn of each fuel in weeks of its cap against the top-weeks the fabs got
   (`an9.py`). Small: TW gas 36.1, crude 22.2, fabs 13.2; KR 40.8 / 31.0 / nuclear 51.3, fabs 16.9; JP 30.1 / 27.0 /
   51.6, fabs 7.7; EU 29.3 / 24.3 / 51.7, fabs 6.3. Full: TW 84.7 / 66.1, fabs 50.1; KR 90.1 / 92.5 / 100.9, fabs
   61.5; JP 85.7 / 73.4 / 102.8, fabs 40.7; CN 82.7 / 54.1 / 94.3, fabs 29.7; EU 95.8 / **37.9** / 95.4, fabs 19.8;
   SEA 72.4 / **27.1**, fabs 11.9; US 103 on all, fabs 49.0. Crude is the scarcest fuel in most episodes of every
   grid but KR on Full.
3. **Why it is not taken.** Not established. Candidates in the code: whole weeks are asked by fuel shares, not by
   exit room (`plan_core.py:555-564`, `603-605`); every wafer away from the rules' plan costs 5 000 USD
   (`regime.json` `anchor.wafer`, `agent.py:764-768`), the size of the lot's own energy cost; whole weeks are asked
   only in the window's weeks 1 to 14 (`hull_until` 12, `agent.py:235`, `823`).
4. **Size.** The 20 bn (Small) is an upper bound on what a forecast "as observed" could justify; the 48 bn is the size
   of the gap to the full-future plan (0.032–0.035 is 35–38 bn). Told the ends of running events, the pool barely
   moves (`an14.py`: base 48.5, ends told 46.1, `w50as` 47.6, while the cost falls 9.9): it is **not** the pool of
   ends. The measured shortage gain of the full-future descent is 16.7 bn on four episodes, so the bound may be loose
   by up to three times. A tenth of the "as observed" part is +0.002 on Small; Full is a few episodes.
5. **Falsifier, no play.** The same accounting on the full-future descent's trajectories (`cycle_lab`, Small 444
   episodes 0–3): if its pool is not lower than the agent's by about its shortage gain, the accounting overstates.
   Then, from the kept logs, whether the weeks the filter flags are weeks the hull did not ask whole. One switch
   after that: `anchor.wafer` 5 000 against 1 000.
6. **Risk.** The bound ignores crude at TW and KR (not a gate there, but 29 % and 57 % of the top), the ration's
   prep week inside a cycle, and exit room being used by chips the plan would sell anyway. Blind bunching is at
   break-even: 13 % of the energy-feasible lots find an exit, against a break-even of 10–16 %.

### 2. Wafers steer the split of a partial top between the fabs of one grid

**Kind:** certain mechanics, uncertain size.

1. **Rule.** `production.allocate_energy` (`production.py:25-27`) shares what is left after the base load pro rata to
   the fabs' requests, and a request is `e * min(alpha R cap, wafers on hand) / R` (`sim.py:322-325`, `349`).
2. **Mechanism.** Lots. In a week with a partial top, a fab with no wafers asks nothing and the other takes its
   share. Chips per GWh: memory 40 mn USD, leading 25, mature 11.5, each times `R` (a damaged fab draws full energy
   for `R` of the lots).
3. **Why not taken.** In `_lots` a fab at capacity on a short grid has `p = rho cap` with the grid's common `rho`
   and the row `wafers on hand >= cap` (`plan_core.py:746-747`, tag `b1_W`): inside the cell it cannot be starved.
   Leaving that regime needs a tie read through the duals, and wafers already at a fab cannot be taken away.
4. **Size.** Energy-bound weeks an episode: TW 17 of 52 on Small and 53 of 104 on Full, KR 30 on Full. Energy going
   to the lower-value fabs in them: Small TW 790 GWh, EU 308; Full TW 5 598, KR 1 507, EU 883. Moved to the fab
   worth more, with the wafers that fab had, and the extra `chip_le` checked against exits with room 12 weeks later
   and charged the mature lots given up: **Small 1.9 bn** (26 of 64 episodes over 1 bn), **Full 4.8 bn** (8 of 16).
   With wafers arranged ahead the exposure is 3 to 5 times that. It draws on the pool of idea 1.
5. **Falsifier.** Counterfactual in the agent's own replay: for a grid-week the plan has partial, the wafer slots of
   the fab with the lower dual per GWh set to zero the week before, kept if the played window is cheaper. Ten
   states with the exact environment snapshot (`rl_lab/counterfactual.py`) would tell first.
6. **Risk.** Where `chip_le` has no exit the right direction is the opposite one; the direction must come from the
   plan's duals, not from list prices. Lean wafer stock at the starved fab is needed weeks ahead.

### 3. A margin of gas at the grid while its output is cut

**Kind:** certain mechanics, zero downside by construction where the terminal has the gas.

1. **Rule.** The observation shows `G_bar` at the instant the week starts; the segment's cap is `share * G_bar` of
   the week's average and the ration reads last week's stock at the grid (`sim.py:337-342`, `356-360`).
2. **Mechanism.** Whole grid-weeks. The program keeps the grid's gas at the threshold exactly (33 % of full-burn
   weeks end within 2 % above it). In the week a shock ends the grid burns more than planned, the stock falls under
   the threshold, and the next week is rationed: the top is lost though the terminal holds gas.
3. **Why not taken.** The cell's row is `stock >= thr (1 - 4e-7)` (`plan_core.py:612-613`, `Fon_prev`) and holding is
   the same on both sides of the valve, so the vertex sits on the threshold.
4. **Size.** Weeks a shock ends: 1.0 an episode on Small, 4.1 on Full. Of them, stock at the threshold, pulled under
   it, gas at the terminal: 0.225 and 0.375 an episode (9 and 6 cases). Next week's shed for it 0.29 and 0.56 bn; lots
   lost (lots two weeks later minus next week's) 0.67 and 2.9 bn: **1.0 and 3.4 bn**, KR first. The lots figure is
   crude. Cases: Small 444 episodes 29 (KR week 5), 39 (KR week 2), 12 (EU week 39).
5. **Falsifier.** One switch in `act` after `_ask_early`: where the observed output of a grid is under nominal and
   the plan keeps its ration at 1, the gas valve is raised by `share * (G0 - G_now)` out of what the terminal holds
   beyond the plan's own use. Identity gate: an episode without a shock to the cent.
6. **Risk.** A plan that wants the grid drained for the off part of a cycle; the rule must not act there.

### 4. The uniform over-ask for tanker releases

**Kind:** certain mechanics, zero downside, not measured (the pickles do not keep override quantities).

1. **Rule.** Overrides pass the same clip as dispatches, edge first, then queue content, then the pool's throughput,
   each pro rata (`chokepoint.py:182-189`). `Episode.actions` writes every tanker release as an override capped at
   the instant's throughput and capacity (`plan_core.py:344-369`); `_ask_scaled` multiplies dispatch slots only
   (`anastasiia_plan_hazard2/agent.py:463-478`), and containers get the week's own capacity by default release.
2. **Mechanism.** Fuel delivered a week sooner in the week a military closure or a cut out-edge of a strait ends.
3. **Why not taken.** Listed as not done in `hazard_lab/README.md`.
4. **Exactly neutral form.** All override slots of a pool at a strait times one number, only when the plan's
   releases use the observed throughput, and the number capped so that no slot exceeds its out-edge's observed
   capacity (then (4) stays idle and (6) returns the plan). For a cut out-edge the plan fills: its slots alone.
5. **Size.** An estimate, by exposure a third to a half of what the edge over-ask took (+0.0009 Small, +0.0019
   Full): fuel through straits is Qatar gas, Gulf crude and Panama cargo, and weather closures are already taken. **Falsifier:** the `look: 1` oracle on fields `o`; and keeping
   `override_executed` in `hazard_lab/play.py` would let the next audit count it.
6. **Risk.** A raised detour release draws on the fleet pool after the clip; neutral while the cut holds.

### 5. Gas throttled beside idle free generation

**Kind:** certain mechanics, measured small.

1. **Rule.** All segments run at one load factor (`sim.py:355-361`): a grid offering more than its load burns fuel
   beside idle free generation. A ration held below 1 at the grid serves the same load with less gas.
2. **Measured.** Gas burned so: Small 2 350 GWh an episode (TW 797, KR 1 143), Full 11 900 (US 5 800). Banked in the
   grid's own stores and matched to that grid's later shed for want of gas: 164 and 174 GWh, **0.7 bn on both**.
3. **Why small.** The program throttles where its forecast gives the gas a use; the rest sits in gas-rich grids.
   As a hedge it costs nothing; 7 of 64 Small episodes hold over 1 bn each.

### 6. Containers held upstream of a throughput-bound strait

**Kind:** speculative.

1. **Rule.** Default release serves arrival cohorts first in, first out and shares the pool pro rata to quantity
   inside a cohort (`chokepoint.py:121-145`); no override exists for containers.
2. **Measured exposure.** Weeks an episode a strait is bound by container throughput with a queue: Small Malacca 5.1,
   Taiwan 1.4; Full Malacca 9.4, Cape 5.9, Hormuz 2.0, Suez 1.8, Taiwan 1.8. At Malacca on Small what passes in those
   weeks is 42 % wafers, 49 % `chip_mat_raw`, 5.5 % `chip_le_raw`, while 121 thousand `chip_le_raw` wait.
3. **Why not taken.** The cell fixes each lane's share of its queue (`plan_core.py:694-708`), so withholding wafers
   shows no gain for the chips behind them.
4. **Size.** Not sized: what moves is a delay, and wafers are plentiful. An upper count of swaps is 133 thousand
   unit-weeks on Small and 576 thousand on Full.
5. **Falsifier.** In the replay of the lot book (`an3.py`), drop the wafer dispatches into a strait in its bound
   weeks and count the `chip_le_raw` that reach a plant earlier and find an exit with room.

## 3. Checked, no exploit (do not repeat)

- **Supply lost at full sources.** 39–77 % of supply is lost at sources (wafers 66–77 %, gas 39–65 %). Lost supply
  an open direct edge could have carried to a grid short of that fuel: 3 GWh an episode on Small, 29 on Full.
- **Gas idle while its grid sheds for want of gas.** Net of the week's arrivals: 59 GWh an episode on Small (0.24 bn).
  A first count of 22 bn was late arrivals.
- **End of the episode under the ration.** Steady feeding maximises total burn; what is left is two weeks of
  arrivals, as FINDINGS says. A final full-burn block leaves more, not less (worked through for EU and TW).
- **A cycle loses no gas.** The prep week burns none because the ration is 0, and the threshold stock burns down in
  the last full week.
- **Chips parked behind a cut edge are surplus.** `chip_le_raw` sent into a queue two weeks deep or more: 123
  thousand an episode on Small, 1.32 mn on Full. Rerouted on the sender's open direct edges they find an exit with
  room for 4.3 and 0.6 thousand (0.22 and 0.03 bn). The 0.93 mn standing at Taiwan at the end of a Full episode are
  two episodes of 16 (8.2 and 8.0 mn).
- **Lots in weeks with energy to spare.** After the check of the wafer source's own stock: 0.33 bn on Small, 0.5 on
  Full; started blindly −0.9 and −12.2. Fabs run at 26–52 % of capacity then for want of wafers at the source or of
  an exit.
- **Plant throughput shared by two raw chips.** Bound 1.4 weeks an episode on Small (`osat_tw`), 7.5 on Full; a
  delay of a week, no chip lost.
- **Markets as a buffer** (4 weeks of demand, equal penalties): nothing to bank, every market is short.
- **Fleet pool.** Counted after the clip on executed quantities; what it scales stays in stock.
- **Override order, FIFO within a cargo, lane tags, `hold`.** No interaction with other cargo (section 1); `hold`
  equals an override of 0.
- **Stock clip.** Reads last week's stock, known exactly; a request above stock only shares it pro rata.
- **The week an event starts.** Requests are scaled pro rata on the edge, the plan's split stands. A common factor on
  all requests from one emptied stock is neutral only if no edge is full.
- **Cargo under way** arrives whatever starts later (`sim.py:304-311`); a container whose next edge is prohibited
  waits (`chokepoint.py:128-131`). Front-running announcements is the `pending` setting, measured −0.0005.
- **Tariffs** are charged at dispatch at that week's rate; nothing to pre-ship, markets are short.
- **Salvage** is 1–18 USD a chip and 500–2 000 a GWh; the gas the plan leaves at straits at the end is worth
  about 20 mn USD.
- **`alpha` above 1** after a fab hit asks for more than the top holds; it only shifts shares (idea 2).
