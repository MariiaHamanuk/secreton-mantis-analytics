# Rollout lab: the simulator inside the agent (7 October)

The foundation of an agent that tries a change of this week's action on its own copy of the simulator before sending it
(a rollout), and of the search lab's third step (`../search_lab/`, a plan carried from week to week inside the agent).
Both need the same three things, and this folder builds and checks them: the simulator's state from the observation,
the week's step, and the observation back from the state.

Inside the agent a rollout does not fit Full's CPU budget (below). Offline the same model is a teacher, and its first
lesson became a rule: **straits as forward stores**, worth +0.006 on Full and +0.011 on Small on top of the hybrid on
episodes nothing was tuned on (formally: +0.011 on Small, +0.002 on Full). The candidate is
`agents/anastasiia_hybrid_hub/` of the repository.

## What is here

- `model.py`: `Model`, for an agent's folder. `window` rebuilds the simulator's state from the observation and takes
  the network "as observed" for the coming weeks; `step` plays a week with the agent's flat action; `flat` writes the
  observation a policy would read after those weeks; `restart` gives another candidate a fresh state on the same
  network. It reads the observation through the hybrid's `lp_part.Planner` and uses the copy of the package's modules
  the hybrid already ships (`sbfv/dynamics/sim.py`): nothing new to copy into a zip. Small and Full only.
- `check.py`: an agent plays real episodes and every week the model is set against the environment: the state, one week
  ahead, the next observation, rollouts of several weeks by a copy of the agent, and the CPU time of each piece.
  `--by_field` repeats the rollouts with one group of the network's fields true: which knowledge the forecast lacks.
- `candidates.py`: does the model rank changes of the week's action as the true future would? Three crude changes (all
  fuel requests times 1.5 or 0.5, no wafers this week) against the base, on the network as observed and on the true
  one, over 8, 16 and 24 weeks, with the shortcuts a real agent would want: an end value after 8 weeks, and the base
  rollout's actions replayed in place of the rules.
- `grid_spells.py` and its `grid_recovery.json`: how long a spell of low output at a grid lasts, from 320 generated
  scenarios of root 333; `Model(..., grids=table)` forecasts the grids' output with it.
- `labels.py`: the model as a teacher. Per decision, grid and fuel: what a change of this week's action is worth over
  24 weeks on the true network (one more week of burn ordered, half the order, more or nothing through the terminal's
  valve), beside the state the fuel rules saw.
- `agents/rules_x/`: `anastasiia_rules_v3` with the switches the labels asked for, every variant a `params.json`
  (`FUEL` in its `fuel_part.py`); the ones that did not help are kept there, switched off, as the record of what was
  tried. The hybrid's variants below are `anastasiia_hybrid_chiplp` with this `fuel_part.py` and a `params.json`.
- `agents/anastasiia_hybrid_hub/` of the repository: the candidate. `anastasiia_hybrid_chiplp` with the two rules
  that helped, in a clean copy of the fuel rules (`hub` and `pipe_cap`, items 6 and 7 of its docstring); the other
  files are the hybrid's.

Outputs of the day (local, not in git): `outputs/rollout_lab/20261007_check/`, `outputs/rollout_lab/20261007_grid/`,
`outputs/rollout_lab/20261007_gap/`, `outputs/rollout_lab/20261007_rule/` (labels, every paired comparison below, the
formal evaluation), `outputs/rollout_lab/agents/` (the variants' folders).

## Results

`agents/anastasiia_rules_v3` plays (the hybrid's linear program is too slow to be a rollout's base policy); root 111;
everything checked by one run. "Share off" is the sum of absolute differences over the sum of the true values.

**The model is the environment's own.** Small, 8 episodes (416 weeks) and Full, 8 episodes (832 weeks):

- the state rebuilt from the observation (stocks, shipments, queues with the order an override takes them in, work in
  process, backlog): equal to the environment's state to rounding (1e-16 of the totals or less), every week;
- one week ahead with the week's true network: every quantity and the cost equal to rounding, every week;
- `flat` after that week: equal to the next real observation in every rebuilt field.

**One week ahead with the network as observed** (what an agent has):

| | Small | Full |
| --- | --- | --- |
| cost of the week, share off | 2.8% | 4.9% |
| of it: unserved demand (demand noise, every week) | 7.4% of that cost | 5.2% |
| of it: shed base load | 1.2%, in 5% of the weeks | 4.9%, in 8% of the weeks |
| lots started | 0.5%, in 5% of the weeks | 0.5%, in 14% of the weeks |
| dispatches executed | 0.06%, in 4% of the weeks | 0.07%, in 10% of the weeks |

**Rollouts of 8 weeks** by a copy of the agent (96 on Small, 104 on Full), the model's cost of the 8 weeks against the
environment's:

| the simulator's network | Small | Full |
| --- | --- | --- |
| as observed | 3.1% off, bias −1.0% | 19.7% off, bias +0.6%; 5% in week 1, 29% in week 8 |
| true | 0.3% | 0.5% |
| as observed, the grids' output true | 1.6% | 1.5% |
| as observed, demand true | 2.3% | 19.0% |
| as observed, any other group true (edge capacity, straits, prohibitions, tariffs, supply, fabs and plants) | 3.0–3.1% | 19.5–19.7% |

- The 0.3–0.5% left with the true network is the copy of the agent reading the first week's network and forecast.
- The forecast "it stays as it is" lacks one thing: the grids' deliverable output. It changes rarely (1–3% of
  grid-weeks; about 3 spells an episode on Full, median 4 weeks, a tenth of them 12 weeks or longer; none announced),
  but a spell at a large grid sheds tens of thousands of GWh a week, so it rules the cost. It hits every candidate
  alike, which is why the ranking below holds up better than these shares suggest.

**The grids' output** (`grid_spells.py`; root 333, 160 scenarios of each network). A spell ends about as likely at
any age: over within 1 week in a quarter of the cases, within 3 weeks in half, within 8 in 80%, within 12 in 90%.
With that table in the forecast ("blend": the expected output; "step": the spell over from the week it more likely
than not is), the model's cost of an 8-week rollout against the environment's:

| the grids' output in the forecast | Small | Full |
| --- | --- | --- |
| stays as observed | 3.1% off, bias −1.0% | 19.7% off, bias +0.6% |
| blend | 2.9%, bias −1.5% | 13.2%, bias −6.9% |
| step | 3.0%, bias −1.5% | 13.5%, bias −6.5% |
| true | 1.6% | 1.5% |

The table takes a third off the error on Full and leaves a bias: the model now expects spells to end and still cannot
see new ones start. Most of what is missing is when a spell begins, which nothing shows.

**Does the model rank changes of the action as the true future would?** (`candidates.py`; Small 16 episodes, 128
decisions; Full 8 episodes, 88 decisions.) Score: the change's cost minus the base's, bn USD per decision; negative
pays. The truth: the true network, the copy of the agent playing its rules every week, 24 weeks. Each row is a way of
scoring the three changes; "picked" is the truth's score of the best of the base and the changes chosen by that way.

| way of scoring | Small: picked (best −1.20) | Full: picked (best −1.78) |
| --- | --- | --- |
| as observed, rules every week, 24 weeks | −0.89; picks a loss in 6% | −0.93; 9% |
| the same with the grids' forecast (blend) | −0.78; 7% | −0.96; 9% |
| as observed, rules every week, 16 weeks | +0.21 | −0.45 |
| as observed, rules every week, 8 weeks | +0.78 | +0.71 |
| true network, 8 weeks | +1.23 | +0.51 |
| as observed, rules for 4 weeks, then the base's actions replayed, 24 weeks | −0.31 | −0.15 |
| as observed, the base's actions replayed from week 2, 24 weeks | −0.10 | −0.08 |
| true network, the base's actions replayed from week 2, 24 weeks | −0.16 | −0.30 |
| 8 weeks and an end value, as observed (odd episodes; best −1.54 and −1.77) | −0.87; a loss in 31% | −0.46; 45% |
| as observed, rules every week, 24 weeks, on the same odd episodes | −1.33; 2% | −0.50; 9% |

- Only a rollout that lets the rules play all 24 weeks ranks the changes: on the network as observed it keeps three
  quarters of what the best pick is worth on Small and half on Full (rank correlation with the truth 0.77 and 0.74).
- A short horizon picks losses even with the true network: "no wafers this week" pays within 8 weeks in most decisions
  (less power to the fabs, less shed) and costs sales later.
- Replaying the base's actions is no substitute for the rules, with the true network either: what the rules do in
  answer to the change is most of its value.
- The end value (a weight per commodity for the stock, the cargo on the way and the lots in process left after 8 weeks,
  fitted on the even episodes) brings an 8-week horizon from a loss to half of the best on Small, with a wrong pick in
  a third of the decisions; on Full, with 44 decisions to fit, it is not usable. Where the data carry it the weights
  make sense: a unit of lng left in the system 3.2 M USD on Small (a unit of shed base load costs 4.1 M), a lot of
  leading-edge chips in process 11 thousand USD.
- The grids' forecast changes the ranking little (within noise on both networks): a spell hits every candidate alike.
- One use of the model as a filter: "fuel x1.5" played always is worth −0.38 bn a decision on Small and −0.29 on
  Full; played only when the 24-week rollout as observed says it pays, −0.8 and −0.9.
- A decision's gain is counted on its own window and the agent then goes on unchanged, so gains of different weeks do
  not add up. For scale, 0.01 of score is 11 bn an episode on Small and about 34 bn on Full.

**CPU seconds per call** (this machine while another lab's search held 8 of its 10 cores; not the container):

| | Small | Full |
| --- | --- | --- |
| the week's window (state and network), once a week | 0.018 | 0.052 |
| a fresh state for another candidate | 0.0001 | 0.0005 |
| the simulator's step | 0.0011 | 0.0044 |
| `flat` | 0.0003 | 0.0009 |
| the rule agent's `act` inside a rollout | 0.0034 | 0.0146 |
| a copy of the rule agent | 0.0034 | 0.0104 |
| a rollout of 8 weeks | 0.038 | 0.16 |
| a rollout of 24 weeks (three times that) | 0.12 | 0.48 |

In the scoring container the rule agent's `act` takes about twice as long (0.03 s on Full, `hub/FORMAL_RESULTS.md`),
so a 24-week rollout on Full is nearer 1 s there: beside the hybrid's 0.9 s, two or three candidates a week fit in
4 s. On Small about a dozen fit in 2 s. Three quarters of a rollout's time is the rule agent's `act`.

## The model as a teacher: straits as forward stores

Root 111 unless said; "+" is the paired difference of score with the base on the same episodes, its 90% interval in
brackets; everything checked by one run.

**What the labels said** (`labels.py`, base `anastasiia_rules_v3`, true network, 24 weeks; Small 16 episodes, 128
decisions; Full 6 episodes, 66 decisions; bn USD per decision, negative pays):

| one more week of lng ordered for a grid that last week | Small | Full |
| --- | --- | --- |
| shed 12% of its output or more | −1.31 (even episodes −2.21, odd −0.61) | −0.67 (−1.48, −0.28) |
| shed nothing | +1.31 | +0.53 |

- By grid, when it had shed: EU −1.75, KR −2.06, TW −1.17, JP +0.30 on Small; TW −1.26, KR −1.14 on Full.
- In the cases that paid most (12 to 24 bn each) the gas was taken from nobody. Small, episode 3, week 9, TW: the rules
  asked for 60 units against a weekly burn of 2,177, because the last edges of TW's three lanes carried 33 of 1,900, 60
  of 855 and 53 of 843; Qatar's source held 20,308 and its edge to Hormuz carried 6,048 with 3,914 asked.
- The order level itself is not what binds: raising or lowering it by the fuel's role (nine variants, `top_by`,
  `need_gain`) moved the score by −0.004 to +0.001 on both networks.
- Half the order costs 2 to 4 bn a decision for lng; nothing through the valve for a week costs 3 to 5.

**From the label to a rule** (base `anastasiia_rules_v3`; Small ×64, Full ×32):

| the rule | Small | Full |
| --- | --- | --- |
| grids that shed are served first in every pass (`short_first`) | +0.0002 (−0.0003…+0.0008) | −0.0006 (−0.0015…+0.0001) |
| a lane may carry what all its strait's out-edges carry, no bound (first 8 episodes) | −0.024 for grids that shed, −0.063 for all | |
| the same until the grid has 2 weeks of burn on the way and waiting (`hub`), grids that shed | +0.0079 (+0.0047…+0.0116) | +0.0008 (+0.0002…+0.0014) |
| ... every grid | **+0.0101** (+0.0064…+0.0144) | **+0.0018** (+0.0009…+0.0028) |
| ... every grid, 4 weeks | +0.0101 (+0.0062…+0.0143) | −0.0007 (−0.0039…+0.0025) |
| ... every grid, 8 weeks, a week of burn at a time | −0.0010 | −0.0053 (−0.0101…−0.0008) |
| ... only lanes cut below half after the strait (`hub_cut`), 2 / 3 / 4 weeks | +0.0094 / +0.0093 / +0.0083 | +0.0017 / +0.0016 / +0.0011 |
| ... only lanes cut below a quarter, 4 weeks | +0.0013 | +0.0001 |
| ... not through a strait already holding a week of its throughput (`hub_strait`), 2 weeks | +0.0101 | +0.0018 |
| ... only grids with fabs (`hub_grids`) | +0.0007 | +0.0008 |
| ... crude too (`hub_fuels`), grids that shed | +0.0079 | +0.0007 |
| a held fuel keeps at the terminal only what arrives in 4 / 8 / 16 weeks (`bank_in`) | | −0.0034 / −0.0016 / −0.0011 |
| a fuel piped straight to the grid is ordered up to 0.9 of the grid's storage (`pipe_cap`) | 0.0000 | **+0.0009** (+0.0008…+0.0010), in 32 episodes of 32 |
| a run or prime week of lng waits while a gating fuel cannot cover the week (`sync_stop`) | −0.0500 (−0.0777…−0.0260) | −0.0018 (−0.0025…−0.0012) |

- Without a bound the rule floods the straits; with one it keeps the cut edge busy every week and has cargo at the
  strait the week the edge passes more. On Small the gain is mostly Europe's lanes through Suez (episode 14: 13.6
  thousand more lng burned at EU, 56 bn less shed).
- A larger stock gains more in most episodes and loses much in a few. Full, episode 3, 4 weeks: 112 bn worse, because
  lng waiting at Hormuz and Malacca leaves ahead of the crude behind it (KR's crude burn 17 to 9 thousand, 3.2 M fewer
  lots at its memory fab). Keeping to lanes that are cut takes that loss to 11 bn.
- On Full, switched on for one grid at a time: EU 2 weeks +0.0010 (better in 29 episodes of 32), KR 2 weeks +0.0008,
  TW 4 weeks +0.0008, IN 4 weeks +0.0006, EU 4 weeks −0.0012; JP, CN, US and SEA never use it.
- Two more hints of the Full labels, tested the same way. Half of the US grid's piped lng order paid in 66 decisions
  of 66: the rules ordered it beyond the grid's storage and the surplus was disposed of; the cap is worth 2.9 bn an
  episode, every episode (Small has no such fuel). Half a week more of banked crude through SEA's valve paid in 33
  decisions of 33, yet passing a trickle of held fuel on every week loses: a label is one step, a rule is every week.

**On the hybrid** (base `anastasiia_hybrid_chiplp`), with the episodes of root 444, on which nothing was chosen:

| | Small 111 ×64 | Full 111 ×64 | Small 444 ×40 | Full 444 ×20 |
| --- | --- | --- | --- | --- |
| the hybrid | 0.8246 | 0.8552 | 0.8566 | 0.8374 |
| 2 weeks, any lane | +0.0103 (+0.0065…+0.0146) | | +0.0101 (+0.0061…+0.0146) | +0.0032 (+0.0009…+0.0057) |
| 3 weeks, cut lanes | +0.0089 (+0.0052…+0.0134) | | **+0.0109** (+0.0066…+0.0156) | +0.0052 (+0.0026…+0.0078) |
| the same and `pipe_cap`: the candidate | as the row above (Small has no piped fuel) | **+0.0019** (+0.0002…+0.0039) | as the row above | **+0.0061** (+0.0034…+0.0087) |

- On Full 111 the candidate is better in 49 episodes of 64 (the first 32: +0.0028, +0.0011…+0.0045; the other 32:
  +0.0010, −0.0020…+0.0041; the worst episode loses 82 bn); on root 444 in 19 of 20 and 30 of 40.
- **The hybrid's score depends on how busy the machine is.** Its linear program has a timer in CPU seconds, and on
  an overloaded machine the same solve costs more of them, so more weeks fall back to the rules. The first run of
  these variants on Full 111 x32 shared the machine with other runs: the base scored 0.8621 where it scores 0.8680,
  and the differences came out too large (+0.005 to +0.008). The table keeps only runs whose base matched its known
  score (0.8552 on Full 111 x64, 0.8374 on Full 444 x20, 0.8246 on Small 111 x64): the Full 111 column of the two
  rows without `pipe_cap` is empty for that reason. Compare the hybrid's variants on a quiet machine and check the
  base's score against a known one.
- Formal evaluation of the candidate before it was committed (`hub/eval/formal_eval.py`; the row read `+dirty`; the
  row of `hub/FORMAL_RESULTS.md` is a separate run of the same zip at its commit): Small 222 x256 0.852 (0.841–0.863),
  +0.0114 (+0.0092…+0.0139) against the hybrid; Full 222 x128 0.855 (0.843–0.868), +0.0019 (+0.0004…+0.0034); in the
  container 0.17 / 0.22 / 0.36 s a week on Small and 0.92 / 1.16 / 1.49 s on Full (median / 95th percentile /
  largest); no week played by the naive rule; zip `9607f89f`. With `hub` alone (zip `b5fe8421`): the same on Small,
  +0.0012 (−0.0003…+0.0026) on Full. So Small's gain holds on every set, and Full's is small: +0.002 on the 128 formal
  episodes and on the 64 tuning ones, +0.006 on the 20 of root 444 (0.8435 against 0.8374).
- A second round of labels with the rules in the base (Small 16 episodes, Full 8): one more week of lng for a grid
  that shed is now worth +0.21 on Small (it was −1.31; EU +1.28, KR −1.10, TW −0.89) and +0.20 on Full (it was
  −0.67); half of the US grid's piped order 0.00 (it was −0.09). What the labels still hinted lost in the paired
  tests: a little more of a gated fuel through the valve (`gate_keep` 2.5 and 2.0: −0.0004 and −0.0014 on Small), and
  JP's lng run paused while its crude is short (`sync_stop`, the row above).

## What follows

1. **Inside the agent on Full a rollout does not fit as it is.** It needs the rules for 24 weeks: 0.5 s here and about
   1 s in the container for each candidate, beside the hybrid's 0.9 s of 4 s. The shortcuts tried (8 weeks, an end
   value, replayed actions) lose most of the value. On Small a dozen such rollouts fit in 2 s by this machine's clock.
2. **Offline it is a teacher, and the loop works.** With the true network the model is the environment (0.3–0.5% of
   cost over 8 weeks), so any decision of any episode can be labelled with what each change is worth, a rollout a
   quarter of a second or less. A label is one step around the base policy, and applied every week it can turn into
   its opposite (the unbounded rule above): read the state the label was made in, write a rule, test it as an agent,
   label again with the rule in the base.
3. Not done: labels on the hybrid as the base and for the chip entries; the truth to the end of the episode instead
   of 24 weeks; lanes with a prohibited edge after a strait as feeders of the strait's stock; a stock per strait
   instead of per grid; the scoring container's clock for the rollouts.

## Beside: where the hybrid loses on Small

`../heur_lab2/tools/account.py --plan --save` and `../heur_lab3/tools/plan_gap.py` on the 40 episodes of root 444, the
hybrid beside the plan that knows the future and serves the base load first (`outputs/rollout_lab/20261007_gap/`):

- 99 bn USD an episode to the plan (v2: 114): shed 53, the same as v2 (KR 17, EU 16, JP 12, TW 8), `chip_le` 38.5,
  `chip_mat` 3, the rest 5. The linear program closed 11 bn of chips and none of the shed.
- The `chip_le` gap comes late: 15 bn in weeks 27–39, 27 bn in weeks 40–52 (the plan sells 0.30 M and 0.54 M more).
- Lots by quarter, the hybrid against the plan: 3.90 against 2.00 M in weeks 1–13 (fab energy 5.2 against 2.7 TWh),
  then 3.31 against 3.14, 2.95 against 3.01, 0.22 against 0.04. The hybrid starts twice the plan's lots in the first
  quarter and sells no more for it; it ships 12.8 M raw `chip_le` out of the fabs (plan 11.3) and 12.7 M packaged
  out of the plants (plan 13.4).
- Skipping all wafers of week 1 is not the remedy: it costs 5.3 bn over 24 weeks (`candidates.py`, Small).
