# strait_wise: rules that keep cargo from going where it cannot pass or be used

Written by the lead from the subagent's hand-back (the subagent could not write files in `MAIN`); its raw records
(every variant's parameters and scores, `diag.py` and `balance.py` output of the final agent, the `sbf check` logs)
are in `strait_wise_lab_out/`. Agent: `outputs/heur_lab/agents/strait_wise/agent.py`. All numbers: Small, root 111.

## Result (64 episodes, paired against `pull`)

| agent | score | vs pull | 90% interval |
| --- | --- | --- | --- |
| pull | 0.4968 | | |
| lane rule alone | 0.6011 | +0.1043 | +0.0824 to +0.1263 |
| + plant balance | 0.6323 | +0.1355 | +0.1100 to +0.1617 |
| strait_wise, final | 0.6523 | +0.1554 | +0.1300 to +0.1819 |

By harm level: 0.560, 0.726, 0.762, 0.717 (pull 0.388, 0.578, 0.624, 0.612). `sbf check` passes on Small (worst week
13 ms) and on Full (worst week 49 ms).

## What it does

`Agent.adjust(flows, observation)` takes any request vector and returns it adjusted; it only lowers or re-routes inside
a (destination, commodity) group. Everything is read from `config`.

1. **Lane bottleneck.** One weighted max-min allocation of all requests over this week's edge capacities
   (`graph_now.u`), strait throughputs (`graph_now.kappa.*`) and the stocks they draw on. A lane gets no more than its
   tightest edge and strait carry; chip_le is served before chip_mat on a shared edge; volume cut on one route moves
   to another route with the same destination.
2. **Plant balance.** Raw chips go to a packaging plant only up to what its open routes to markets can ship on, and
   the room in its storage. Export controls often leave a plant with almost no route out for the whole episode
   (episode 10: osat_kr can ship only to Japan, fills its storage by week 4 and disposes of 185,000 chip_le a week).
3. **Fab storage.** Wafers into a fab up to its storage less stock and cargo on the way.
4. **Last weeks.** Chip-chain cargo that cannot be sold or started before week T is dropped; the time needed is read
   from the network (14 weeks for a leading-edge wafer on Small).
5. **Fuel cover.** Fuel into a grid only up to what it can still burn before the end. On Small every grid starts with
   52 weeks of nuclear fuel, so none is sent (`pull` pays about 6 bn USD per episode shipping it).

## What mattered (leave one out of the final agent, 32 episodes)

| left out | change | 90% interval |
| --- | --- | --- |
| plant balance | -0.0317 | -0.0444 to -0.0205 |
| stock rows, reroute, priority fill | -0.0069 | -0.0116 to -0.0027 |
| fuel cover | -0.0054 | -0.0072 to -0.0038 |
| fab storage | -0.0050 | -0.0062 to -0.0041 |
| last weeks (chips) | -0.0037 | -0.0050 to -0.0025 |
| queue ramp | -0.0014 | -0.0045 to +0.0011 |

The lane rule is the base and is worth +0.10 to +0.12 on its own.

## Tried and dropped

- Tariff preference between routes: -0.005 to -0.032 (a tariff hits every route between two regions alike).
- Tanker release overrides at straits, in proportion to what each lane can carry: -0.032; only for blocked lanes: 0.
- Cutting lng and crude in the last weeks: adds shed.
- A plant allowed to fill 2x or 4x its storage: -0.033; 0.8x or 1.2x: -0.004.
- The simulator's fleet slack as rows of the allocation: -0.001, not significant.

## Where the rest of the gap is

Costs per episode (12 episodes): queue holding, disposal, tariff and freight fall from 27.5 to 9.3 bn USD (clairvoyant
5.8); shed 2,083 against the clairvoyant's 2,052; shortage 1,051 against 747. Demand served equals the clairvoyant
plan's in weeks 1 to 12 and is half of it from week 13 on: what is left is fab output, that is fuel priority to the
grids whose fabs can run.

## Corrections to the lab README

- The clip drops a slot only when its own first edge is prohibited; `action_mask` is stricter (any edge of the lane).
  A lane slot with a later prohibited edge is executed and its cargo waits at the strait for good: multiply by
  `action_mask`.
- The simulator also has a fleet slack: flows on duplicate sea routes (Cape, Lombok, east of Taiwan) are scaled down
  together when their extra transit exceeds a fleet cap.
- Storage, holding cost, salvage and supply are under `instance["nodes"][i]["stock"][commodity]`.
- `graph_now.u`, `.open`, `.kappa.*` are the values at the end of last week, not this week's averages.
- An override release takes the oldest units of the commodity at the strait whatever their lane.
