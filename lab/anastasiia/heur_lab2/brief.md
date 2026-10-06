# Heuristics lab 2: making the rule agent better with what the simulator does by itself

Goal: raise the score of the team's rule agent, `agents/anastasiia_rules_fuelchip` (no linear program in `act`).
It scores 0.797 on Small and 0.780 on Full; a plan that knows the future and obeys the simulator's rules scores about
0.95 on its own cost and 0.87 when its orders are replayed blindly. The gap is understood now (below): close a part of
it.

`MAIN` is `/Users/anastasiiamazur/Projects/secreton-mantis-analytics` (the team's checkout). You work in your own git
worktree made from its last commit: it holds the agent, `hub/` and `docs/`, but **not** `outputs/` (this lab) and no
`.venv`. Read this lab from `MAIN` by absolute path.

## Rules of the lab

- **Never run `uv run` or `uv sync`**: they would build a second environment and miss the cache of reference costs.
  Call the team's interpreter by absolute path: `PY=$MAIN/.venv/bin/python`, `SBF=$MAIN/.venv/bin/sbf`.
- **CPU: at most 2 worker processes at any time** (`--n_jobs=2`, one command at a time). The machine is shared with
  other people's long experiments whose results depend on its speed. Never start two evaluations in parallel.
- Write only inside your worktree, plus `MAIN/outputs/heur2/agents/<your name>/` (your final agent folder) and
  `MAIN/outputs/heur2/reports/<your name>.md` (your report). Touch nothing else in `MAIN`: no edits to `agents/`,
  `hub/`, `src/`, `team/`, `lab/`; no git commands in `MAIN`; no commits anywhere; `sbf upload` is forbidden.
- Episodes: decisions on **root 111** (`--entropy=111`), 32 episodes while exploring, 64 to confirm. Diagnostics may
  also use **root 444, episodes 0..39** (the only ones with the best plan to compare with). Never use roots 0, 222, 333.
- An agent may import only the standard library, numpy, scipy and torch, and must read every size and table from
  `config` (it is scored on Full too: 8 grids, 16 markets, 104 weeks): no slot numbers, node names, commodity names or
  capacities typed in. A week whose `act` raises is played by the naive rule: never wrap code in a bare `except`.
- Keep your change small and local (new functions, a few call sites), with parameters in the `PARAMS` / `FUEL`
  dictionaries and a switch that turns the new rule off: three people change the same agent and the lead merges.

## Tools

```bash
MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics; PY=$MAIN/.venv/bin/python; LAB=$MAIN/outputs/heur2
BASE=$MAIN/agents/anastasiia_rules_fuelchip
# the decision tool: paired comparison with the base agent (about 2 minutes per agent on 64 episodes)
$PY $MAIN/hub/eval/compare.py --base=$BASE --episodes=32 --n_jobs=2 /abs/path/to/agents/mine [/abs/another]
# where an agent loses: lots, disposal, power, fuel, from the simulator's own records; --plan: beside the best plan
$PY $LAB/tools/account.py /abs/path/to/agents/mine --episodes=16 --n_jobs=2
$PY $LAB/tools/account.py /abs/path/to/agents/mine --entropy=444 --episodes=40 --plan --n_jobs=2
# the network: grids (shares, burn, thresholds, storage), fabs, plants, markets, every action slot with its route
$PY $MAIN/hub/eval/routes.py small        # or full
# the server's check; run it on the final agent on both networks
$SBF check /abs/path/to/agents/mine --task=small ; $SBF check /abs/path/to/agents/mine --task=full
```

A difference is real only when the 90% interval of `compare.py` excludes 0. `account.py --save=x.npz` keeps the weekly
arrays (lots, energy, disposal, stock, shed, segment, sent, asked per episode) for your own analysis;
`$LAB/data/base_444.npz` is the base agent's on root 444, `$MAIN/outputs/plan_stats/20261006_040041/episodes.npz` the
plan's (`plan_lots`, `plan_sent`, `plan_stock`, `plan_shed`, `plan_served`, `plan_fab_energy`, `capacity`, `banned`).
`$LAB/reports/base_444.txt` is `account.py --plan` for the base agent.

## The agent you start from

`$BASE/agent.py` combines two parts, each filling its own slots of `flows`:

- `fuel_part.py`, `FuelRules.fill(observation, flows)`: lng, crude, nuclear fuel. Terminal-to-grid valve; time
  concentration of the rationed fuel (`_hold`: on / run / hold / prime); order-up-to orders from the sources within
  what lanes carry, grids in priority order, in stages (`share_first`, strict priority, top-up).
- `chip_part.py`, `Agent.fill_chip_flows(observation, flows)`: wafers (`_wafer_flows`: 3 weeks of capacity at each fab,
  scaled by `usefulness`), raw chips (`_raw_flows`: a small max-flow fab -> plant, plants filled by expected wait),
  packaged chips (`_pack_flows`: max-flow plant -> market by need).

Copy the folder to `agents/<your name>/` in your worktree and change the copy. Field reference: `docs/fields/small.md`,
`docs/GUIDE.md`. What was tried before and its verdicts: `hub/tried/heuristics.md` (read it: do not repeat).
What the team knows: `hub/FINDINGS.md`.

## What the simulator does by itself (its code: `$MAIN/.venv/lib/python3.13/site-packages/shockbench_flow/dynamics/`)

The agent only decides what to ship on each route. Everything else is automatic, every week, in this order:

1. Straits release queued cargo, oldest first, within throughput.
2. Requests are clipped (edge capacity pro rata, then last week's closing stock pro rata); a request that fits is
   executed exactly. Then arrivals.
3. **Grids** (`sim.py`, step 7). For each fuel `k`: `av_k = min(share_k * G_bar * ration_k, stock on hand)`, where
   `ration = min(1, last week's closing stock / threshold)` for the rationed fuel (lng) and 1 for the others. The
   no-fuel segment `av_null = share_null * G_bar` is always there. `g_av = sum av_k + av_null`.
   Each fab asks `E_hat = e * min(capacity, wafers on hand)`. Base load first: `y = min(base load, g_av)`; the fabs
   share `min(sum E_hat, g_av - y)` pro rata to what they ask. **Every segment then runs at the same load factor**
   `load = (y + fab energy) / g_av`: fuel `k` burns `av_k * load` and the free segment gives only `av_null * load`.
4. **Fabs** start `min(capacity, wafers on hand)` lots as far as their energy goes. They cannot be told to wait: a
   wafer at a fab is started the first week there is power. Lots come out `tau` weeks later as raw chips **into the
   fab's own stock**; stock above the storage is disposed of.
5. **Plants** package `min(throughput, raw stock)` at once; packaged chips come out 2 weeks later into the plant's
   stock. Stock above storage is disposed of.
6. Markets serve demand from stock; unserved demand is lost (about 50,400 USD per `chip_le`, 10,300 per `chip_mat`).

Consequences:

- The sliver above the base load (`G_bar - base load`: 7.0% of output at TW, 3.5% at KR, 1.0% at JP, 0.4% at EU on
  Small) is all the fabs ever get. A week's fab energy is `clip(sliver - sum of the fuels' shortfalls, 0, what the fabs
  ask)`, each fuel's shortfall being `share_k * G_bar - av_k`. This is convex in the shortfall: a fuel whose shortfall
  can exceed the sliver should come in full weeks and empty weeks, not as a steady partial flow.
- **A grid that offers more than the load takes loses free energy.** Complete grid, fabs asking nothing: `load` =
  base load / output, every fuel burns `load` of its cap and the free segment (43 to 58% of output) gives `load` of
  its own. With the gas ration held at `rho* = 1 - (sliver - fab ask) / (gas share * G_bar)` instead, the free segment
  runs at 100% and the gas saved is `share_null * (sliver - fab ask)`: 221 GWh a week at TW, 165 at KR, 103 at JP, 123
  at EU when the fabs are idle (1,000 GWh = 4.1 bn USD of base load).
- Lots can only be timed by when wafers reach the fab, power by when fuel reaches the grid.

## What the base agent loses (root 444, 40 episodes; `reports/base_444.txt`)

Gap to the best plan's own cost: **141.6 bn USD per episode = chips 66.3 + shed 68.0 + other 7.3** (naive minus
clairvoyant is about 1,077 bn: 10.8 bn is 0.01 of score).

1. **It makes chips nobody can take.** Fabs dispose of 1.79 M `chip_le_raw` and 0.87 M `chip_mat_raw` per episode
   (KR 0.80, JP 0.60, TW leading 0.26, EU leading 0.13; EU mature 0.65, TW mature 0.22); the plan about 0.16 M. The
   plan starts 1.28 M fewer lots at KR and 0.47 M fewer at TW leading than the agent in the episodes where it starts
   fewer, and in several episodes none at all (episode 11: 0.11 M against 7.67 M, the same chips sold): there the
   plant-to-market routes carry so little that the initial work in process alone fills them. Energy the agent spends
   on lots: 13,813 GWh per episode against the plan's 11,043. `usefulness` is a max-flow of each fab **alone**: fabs
   share plants and plant-to-market routes, so every fab looks useful.
2. **It sells 1.21 M fewer `chip_le`** (12.73 M against 13.94 M). The plan ships 0.71 M more raw chips out of the JP
   fab (whose only plant is `osat_my`) and starts 0.39 M more lots at JP in the episodes where it starts more (worth
   19.4 bn), 0.26 M more at KR (12.9 bn), 0.19 M at TW leading (9.4 bn). The agent keeps no raw chips at plants
   (0 on average); the plan keeps 119,000 at `osat_my`, 58,000 at `osat_kr`, 48,000 at `osat_tw`.
3. **JP's sliver is gated by crude.** Crude is 5% of JP's output and the sliver 1%: 20% short of crude and no lot
   starts. Episode 3: JP gas at 100% every week, crude at 65 to 98%, lots 0 to 130,000 a week of 144,000; the plan runs
   JP complete in 36 weeks (the whole 220 bn gap of that episode). In `fill`, stage 1 (`share_first` 0.5) gives KR's
   gas half its need of the strait they share before JP's crude gets its second half. Limiting fuel by episode
   (least weeks of full burn): JP crude 25 of 40, lng 16; TW crude 30, lng 14; KR crude 21, lng 21. The agent
   concentrates only lng in time; crude waits at the terminal only while the lng is on hold.
4. **It loses free energy: 6,193 GWh per episode (25.5 bn)**, the plan none. In weeks without shed the fabs start
   0.56 (TW leading), 0.45 (TW mature), 0.60 (KR), 0.78 (JP) of capacity: the grid is complete, the fabs ask less,
   every segment is throttled. The same happens in the last 14 weeks, when no wafers are sent any more.
5. Fuel logistics: the plan burns 100 GWh a week more lng (KR, EU) and 45 more crude (JP) with less delivered; the
   agent leaves 6,065 units waiting at straits at the end. Not understood yet.

## What to hand in

1. Your best agent folder copied to `MAIN/outputs/heur2/agents/<name>/` (`agent.py`, `fuel_part.py`, `chip_part.py`).
2. `MAIN/outputs/heur2/reports/<name>.md`: what you changed (the rule in plain words, the functions touched); a table
   of **every** variant you tried with its `compare.py` line against the base (score, difference, interval, episodes),
   including what did not help; `account.py` sections that moved, before and after; what you would try next.
3. Your final message: the best paired difference with its interval on 64 episodes of root 111, the difference on
   Full (`--task=full --episodes=16`), whether `sbf check` passes on Small and Full, the one or two rules that
   mattered, and anything in this README that turned out wrong.
