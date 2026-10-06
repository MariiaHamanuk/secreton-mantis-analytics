# Heuristics lab: rule-based agents for ShockBench-Flow

Goal: a rule-based agent (no linear program solved in `act`) with the highest score on Small. The team target is 0.75.
Baselines on Small, root 111, 64 episodes: `template` 0.414, `pull` 0.497, `mpc` (a weekly LP) 0.689.

`MAIN` below is `/Users/anastasiiamazur/Projects/secreton-mantis-analytics` (the team's checkout). You work in your
own git worktree, which holds only the upstream starter kit: `agents/pull`, `agents/mpc`, `team/` and this lab are
**not** in it, read them from `MAIN` by absolute path.

## Rules of the lab

- **Never run `uv run` or `uv sync` in your worktree** (it would build a second environment and miss the team's cache
  of reference costs). Always call the team's interpreter by absolute path:
  `PY=$MAIN/.venv/bin/python`, `SBF=$MAIN/.venv/bin/sbf`.
- Write only inside your worktree, plus two places in `MAIN`: `outputs/heur_lab/agents/<your agent name>/` (your final
  agent folder) and `outputs/heur_lab/reports/<your agent name>.md` (your report). Touch nothing else in `MAIN`: no
  edits to `agents/`, `team/`, `src/`, no git commands there, no commits, no uploads (`sbf upload` is forbidden).
- Episodes: **only root 111** (`--entropy=111`), at most 64 episodes. Never evaluate on roots 0 (dev), 222, 333, 444.
- CPU: at most 3 worker processes at a time (`--n_jobs=3`). Other agents share the machine.
- An agent may import only the standard library, numpy, scipy and torch. It must read every size and every table from
  `config` (it will also be run on Full, a larger network): no slot numbers, node names or capacities typed in.
  `config["static"]["instance"]` holds the whole public instance as JSON (nodes with their `grid` / `fab` / `osat`
  attributes, edges, lanes, stock slots, initial state, `params`); `config["layout"]` says what each row of a dense
  observation block is.
- Seed any randomness from `config["policy_seed"]`. A week whose `act` raises is played by the naive rule: wrap
  nothing in a bare `except`, fix the error.

## Tools (all read-only)

```bash
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics; PY=$MAIN/.venv/bin/python; LAB=$MAIN/outputs/heur_lab
# paired comparison with a base agent (the decision tool): about 1 minute per agent on 64 episodes
$PY $LAB/tools/compare.py --base=$MAIN/agents/pull --episodes=64 /abs/path/to/agents/mine [/abs/path/to/another]
# where an agent loses, physically: costs by component, shed per grid, lots per fab, demand served, flows per commodity
$PY $LAB/tools/diag.py /abs/path/to/agents/mine --episodes=12 --n_jobs=3          # add --nobest to skip the LP (faster)
# the network: grids, fabs, markets, every action slot with its route and lead time
$PY $LAB/tools/routes.py small
# the server's check (imports, timing, shapes); run it on the final agent
$MAIN/.venv/bin/sbf check /abs/path/to/agents/mine --task=small
```

A difference is real only when `compare.py`'s 90% interval of the difference excludes 0. Use `--episodes=32` while
exploring and 64 to confirm. Field reference: `docs/fields/small.md` and `docs/GUIDE.md` in your worktree.
`$MAIN/agents/pull/agent.py` is the rule agent to start from (copy it into your worktree's `agents/<name>/`).

## How the simulator turns requests into flows (read from its code, `$MAIN/agents/mpc/sbfv/dynamics/`)

Every week, in this order:

1. **Straits.** Cargo that reached a strait queues there. It is released up to the strait's throughput
   (`graph_now.kappa.*`, which falls with `graph_now.open`) and the next edge's capacity, oldest arrivals first, pro
   rata inside a week's arrivals. For tanker cargo (lng, crude) `release_mode` 1 with `override_qty` sets the release
   per override slot, and 2 holds it.
2. **Clip of your `flows`.** (a) A slot whose route is prohibited is dropped (`action_mask` 0). (b) Edge cap: if the
   requests on one edge (all commodities, all lanes starting on it) add up to more than `graph_now.u[edge]`, all of
   them are scaled down by the same factor. (c) Stock: if what is then drawn from one stock (the edge's tail node, the
   commodity) exceeds **last week's closing stock** (`stock.qty`), every draw on it is scaled down by the same factor.
   So requests are weights only when something binds: **a set of requests that fits the edge capacities and the stock
   is executed exactly as asked.** "Send the maximum" lets capacity ratios decide who gets a scarce stock.
3. **Arrivals**, then **supply**: each source refills to its storage with at most its weekly supply (unused supply is
   lost). What arrives or is produced this week can be dispatched next week at the earliest.
4. **Grids.** For each fuel `k` of a grid: output available = min(share_k x G_bar, fuel stock on hand); for the
   rationed fuel (lng) the first term is multiplied by min(1, last week's closing stock / (psi x ibar_k)). The no-fuel
   share is always available. Then, with priority `base_first` (every grid here): **base load is served first**;
   only what is left above the base load powers the fabs. Unserved base load is shed at VOLL (4.1 M USD per GWh).
   Fuel is burned pro rata to the load actually served.
5. **Fabs.** A fab wants to start min(alpha_bar x R x cap0, wafers on hand) lots; it starts the share of them its grid
   has power for (`e` GWh per lot). Lots come out `tau` weeks later as raw chips.
6. **Packaging plants** start min(throughput, raw stock), pro rata over the raw chips when short; packaged chips
   come out 2 weeks later.
7. **Markets** serve demand from stock on hand; unserved demand is lost (penalty per unit: about 50,400 USD for
   chip_le, 10,300 for chip_mat). Stock above a node's storage is disposed of at a cost.

## The fact that matters most

On Small each grid's base load is 93.0% (TW), 96.5% (KR), 99.0% (JP) and 99.6% (EU) of its deliverable output, and
the fabs' full need is exactly the rest. **A fab runs only on the top sliver of its grid's output**: a grid at 93% of
output (TW) or 96.5% (KR) starts no lots at all. One GWh in that sliver is worth 6 to 10 times VOLL (a lot of chip_le
is 50,400 USD and takes 0.00126 to 0.002 GWh). So the value of fuel is convex: **filling one grid completely is worth
more than spreading the same fuel over several grids**, and a grid that cannot be completed this week should give its
scarce fuel to one that can.

The clairvoyant plan (score 1) is a relaxation: it powers fabs while shedding base load, which the simulator never
does. 54% of its lots are started that way, so its flows are not a plan to imitate for fuel, and a score of 1 is out
of reach for any agent. `diag.py` prints, per fab, how much of the clairvoyant plan's output is of that kind.

What each grid needs per week at full output, and where it can come from (`routes.py` prints the slots):

| grid | lng | crude | nucfuel | lng routes | crude routes |
| --- | --- | --- | --- | --- | --- |
| TW | 2,177 | 109 | none | Qatar via Hormuz (2 lanes), US via Panama | US via Panama only |
| KR | 2,750 | 220 | 3,300 | Qatar via Hormuz (3 lanes), Australia | Gulf via Hormuz (4 lanes) |
| JP | 5,400 | 900 | 1,440 | US via Panama, Australia, Russia | Gulf via Hormuz (4 lanes) |
| EU | 8,840 | 1,040 | 11,960 | Qatar via Suez or the Cape, US, Russian pipe | US |

- Each grid starts with 52 weeks of nuclear fuel, 1.4 to 2 weeks of lng and **no crude** (it sits at the terminal:
  the terminal-to-grid slots have lead time 0 and must be used every week).
- lng is rationed: once a grid's lng stock closes a week under psi x ibar (TW 2,053, KR 2,357, JP 6,480, EU 10,608)
  its lng output falls in proportion. Keep the stock above it.
- Under `pull` (4 episodes of root 111): grid_tw sheds in 78% of weeks and its lng is under the threshold in 88%,
  while the clairvoyant plan has TW shedding in 28% of weeks, almost none of its TW lots started while shedding. TW
  and KR are the grids where fuel priority should pay first.

Where `pull` loses to the clairvoyant plan (USD bn per episode, 4 episodes): shortage 1,156 against 778, shed 2,397
against 2,167, tariff 15 against 7, disposal 11 against 0.5, queue holding 7.5 against 0.2. `pull` asks for 656,000
wafers a week and the clairvoyant plan ships 159,000; it ships 137,000 chip_le to markets against 272,000.

Other facts measured on 2,000 episodes (`$MAIN/team/stats/event_stats_small.md`): most disruptions are already in
force in week 1 and stay to the end (prohibitions, capacity cuts to 0.25 or 0.06 of an edge's capacity); announced
sanctions are false alarms half of the time; `warning.score` carries almost no information; demand is nearly constant
(11% noise, no trend), so the forecast in the observation is all there is to know about it.

## What to hand in

1. Your best agent folder copied to `$MAIN/outputs/heur_lab/agents/<name>/` (`agent.py`, and `params.json` if any).
2. `$MAIN/outputs/heur_lab/reports/<name>.md`: what the agent does (the rules, in plain words); the table of every
   variant you tried with its `compare.py` line against `pull` (score, difference, interval), including the ones that
   did not help; `diag.py`'s sections 2 to 4 for the final agent; what you would try next.
3. Your final message: the best score against `pull` with its interval, the one or two rules that mattered, and
   anything about the simulator you found that this README gets wrong.

## Corrections found by the first three lab agents (2026-10-06)

- The clip drops a slot only when its **own first edge** is prohibited; `action_mask` is 0 when any edge of the lane
  is. Cargo sent on a lane with a later prohibited edge is executed and waits at the strait for good: always multiply
  by `action_mask`.
- `graph_now.kappa.*` already includes `graph_now.open`: using both counts a closure twice.
- `graph_now.u`, `.open`, `.kappa.*` are the values at the end of last week; the clip uses this week's average. They
  differ only in a week an event starts or ends.
- The simulator also has a **fleet slack**: flows on duplicate sea routes (Cape, Lombok, east of Taiwan, bypass,
  turn-back) are scaled down together when their extra transit exceeds a fleet cap (4,132 a week for tankers on Small).
  "Send the maximum" is always far over it; planning at exactly 100% of it is best.
- An override release takes the oldest units of the commodity at a strait whatever lane they queued for, and sends
  them on the override slot's lane: it re-routes queued tanker cargo.
- Storage, holding cost, salvage and supply are under `instance["nodes"][i]["stock"][commodity]`. Tiny has no
  `layout["lot_keys"]`.
- The convexity of fuel holds in **time** as well: a grid whose lng inflow is below its burn should run complete for a
  stretch and be off for a few weeks (fuel waiting at the terminal), not burn a steady partial amount.
- The plan that obeys base load first scores about 0.95 on its own cost and about 0.9 when its orders are replayed, so
  the reachable ceiling is about 0.9, not far below it.
- `compare.py` takes 3 to 4 minutes per agent on 64 episodes when the machine is shared.

## Result of the first round

`rules_v2` (fuel by `fuel_first`, the chip chain by `chip_route`), now `MAIN/agents/anastasiia_rules_fuelchip/`: 0.782 on Small
root 111 (64 episodes), 0.813 on root 222 (64 episodes, never tuned on), 0.789 on Full root 111 (32 episodes). `mpc`
on the same sets: 0.689, 0.723, 0.423. Passing the chip flows through `strait_wise.adjust` as well gives 0.756, and
`strait_wise`'s own chip rules with the same fuel rules 0.750.
