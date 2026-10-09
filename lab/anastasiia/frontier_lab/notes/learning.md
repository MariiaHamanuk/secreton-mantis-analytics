# Brainstorm, lens "learning and offline compute as a new type of solution"

Working notes, 10 October. Read-only pass over `hub/FINDINGS.md`, `hub/tried/rl.md`, `hub/tried/mpc.md`,
`lab/anastasiia/paradigm_lab/{SPEC,PLAN,BOARD}.md`, `lab/anastasiia/hazard_lab/{README,REVIEW,GENERATOR}.md`,
`lab/anastasiia/cycle_lab/README.md`, the agent `agents/anastasiia_plan_hazard` and the generator in the installed
package. One process of 45 s was run (section 9); nothing in the repository was changed. "Measured" means a number
from the repository with its source; "estimate" means mine and unverified.

Base: `anastasiia_plan_hazard`, 0.899 Small / 0.902 Full (root 222).

## 0. What kind of signal can be learned here at all

Four facts decide every idea below. Each is measured in the repository.

1. **A label must be an expectation over futures, not one future.** The multiplier RL had exact advantages and still
   memorised episodes, because the useful move was decided by the scenario's future (`hub/tried/rl.md`, "Ворота 2").
   So either the learned object is a small table per *context* averaged over thousands of states (the average over
   states of one-future labels is the context's expectation), or the label is made from several futures of the same
   state (idea 3). A per-state function approximator on one-future labels is the dead form.
2. **The closed-loop planner has a chaos floor.** A change with no content (a price of 1e-5 USD on every column) moves
   one episode with a spread of about 0.009 of score, i.e. about 10 bn USD on Small (`hub/tried/mpc.md`, P11,
   placebo row); true information loses 13 to 21 bn in single episodes (`hazard_lab/REVIEW.md`, B1). An "exact"
   advantage with the planner as the tail policy is exact and still carries this noise: about 10 bn against signals of
   0.5 to 5 bn. Consequences: per-state labels need the rules as the tail (deterministic, 0.04 s an episode) for
   screening and the planner tail only in aggregates of hundreds of states; anything that selects per episode among
   planner variants learns the chaos.
3. **The model's own cost is not a judge** (`paradigm_lab/PLAN.md`, section 1). Every learned object below enters as
   a generator (a price, a forecast, a request transform) with an identity gate at zero, and is accepted only by
   paired play on fresh scenarios.
4. **A dollar is not a dollar across harm levels.** The board has equal counts per level and weights 50/30/15/5, with
   similar room per episode (1 061 bn at level 1, 878 at level 4; `hub/FINDINGS.md`, "Рівні шкоди"). A dollar saved
   in a level-1 episode is worth about ten times a dollar in a level-4 episode, and level 1 holds 52 % of the gap to
   the full-future plan (`paradigm_lab/SPEC.md`). Every fit below should weight a training episode by
   w_level / (n_level x room_level), and every mining pass should read level-1 episodes first.

Pools the ideas draw on (Small unless said):

| pool | size | source |
| --- | --- | --- |
| window teacher (same agent, true future in the 26-week window) | +0.017 to +0.026 | task statement; `hazard_lab/REVIEW.md` E |
| best executable full-future plan found | +0.032 to +0.035 (local optimum, a lower bound) | `paradigm_lab/BOARD.md`, P7 |
| ends of running events over `hull3` | +0.0090, of which +0.0050 to +0.0059 taken | `hazard_lab/README.md` |
| onsets of new events (perfect knowledge, from `plan_hull`, 8 episodes, stale) | +0.003 Small, +0.011 Full | `hub/FINDINGS.md` |
| whole-week set family from a fixed state | +0.0018 Small; cheap neighbourhood on Full empty | `cycle_lab/README.md` E2, E6 |
| not information: full-future plan minus window teacher | 0.006 to 0.018 (difference of two ranges; the split 0.026 / 0.009 is unconfirmed) | `BOARD.md`, P7 |
| price terms found blind by the LLM author | +0.0021 Small, +0.0016 Full | `hub/tried/mpc.md`, P11 |

## 1. Mechanism mining at scale: two teachers, action diffs, and belief fuzzing

**Formulation.** Three instruments, one table as the output.

- *Diff pass.* Play the model on fresh scenarios. At every week deep-copy the agent, hand the copy the truth
  (`tell_truth`, switches `truth` / `truth_events` in `hazard_lab/src/agent.py`) and let it plan the same state with
  the same memory. Record per slot: kind (order, valve, wafer, raw, pack, strait release), commodity, the two
  requests, what bounds the model's request (edge capacity of the observed instant, strait throughput, stock, fleet
  pool, storage room), the event on the slot's route (type, depth, age, true weeks to its end, true weeks to the next
  onset), the destination grid's state (shed last week, cover in weeks by fuel segment), and the dual of the slot's
  binding row from the model's own solve. Second teacher: the offline full-future descent (`regime_lab/plan.py
  starts --start_agent`), aligned by week on the same episode; its diff minus the first teacher's is what is *not*
  window information (objective, horizon, optimisation).
- *Clustering.* Cells of (slot kind) x (binding bound) x (event type, age bucket) x (direction of the diff). Rank by
  sum of |diff| x dual, by frequency, and by a flag "the deviation is free if the belief is wrong" (the request is
  clipped by the simulator before stock is drawn).
- *Belief fuzzing* (no oracle needed). For every observation field that is a snapshot of the instant t-1 while the
  simulator applies the week's own value (`graph_now.u`, `.open`, `.kappa.*` are documented; `supply.avail`, `G_bar`,
  `fab.R`, `osat.R`, `prohibited` are to be checked the same way in `dynamics/`), and for every state where the
  plan's request sits on that bound: multiply the requests on that bound only, restore the snapshot (19 ms), play the
  rules to the end (0.04 s on Small), record executed flow and exact advantage. `cycle_lab` measured the blanket
  version dead (-0.0913): raising pays only where the plan was held back by its own false belief, so the unit of
  search is the belief, one bound type at a time.

Data: 256 Small and 64 Full fresh episodes give about 13 000 and 6 600 states; the diff pass costs one extra plan a
week, about 5 and 10 CPU-hours (estimate). Attribution of the top clusters with the planner as the tail needs about
2 000 states a cluster because of the chaos floor: 40 CPU-hours a cluster on Small, cloud only on Full. The cheaper
route skips attribution: write the rule a cluster suggests and judge it as any candidate.

**Why the information is in the state.** The diff pass does not ask the model to predict the teacher. It asks which
of the teacher's deviations could be reproduced without the teacher's knowledge: deviations with a free downside
(clipped requests), deviations whose trigger is observable (depth, age, kind of the cut), and deviations of the
second teacher that the first does not make (no information involved by construction). The early request was exactly
such a cell: (dispatch, bound = instant capacity, stoppage, any age, teacher asks more).

**Why it is not a dead idea.** It is not imitation (P16 fitted a model to the LP's choice and kept 65 to 68 %); no
function is fitted to the teacher. It is not `graft.py` (whole-episode open-loop transplant of one channel, negative
because the gap is in coordinated decisions): here both plans start from the same state and memory each week. It is
the procedure that produced +0.0028 / +0.0014 (early request) and +0.0009 / +0.0019 (uniform multiplier) on 9
October, run over all slot kinds instead of the one a person looked at.

**Size (estimate).** +0.003 to +0.006 on each network, as two or three more mechanics of the size already found.
Pool: what is left of the window teacher after `plan_hazard` (about +0.012 to +0.020) plus the non-information
difference of the second teacher.

**Cheapest falsifier.** The `look: 1` oracle (`truth_events` with `look` 1, already in the code, never run): truth
only for changes within the coming week. It is the ceiling of everything a request of the action week can take.
Under +0.002 over `plan_hazard` on Small 444 x64 and Full 444 x16 means the action-week cells are exhausted and the
mining has to read plans of weeks 2 and later only. Second falsifier, no play at all: the share of the diff mass
(|diff| x dual) that falls into cells with a free downside or an observable trigger; if under 10 %, the teacher's
advantage is foresight and nothing else.

**Online CPU and overfitting.** A mined rule is a few lines before or after the solves, no new solve. Mining on a
new root (say 1001), each rule judged on fresh episodes of another root (1002) with the P11 protocol, confirmed
under the meter on a third, root 222 once.

## 2. Dual audit, then a learned value: P2 and P8 with exact finite-difference labels

**The suspect, read from the code.** The window-end credit is one constant: `end_fuel` 3.3e6 USD a unit for every
fuel of every grid, capped a pool at the grid's stock level plus `end_weeks` (1) of burn and at what the grid can
still burn (`agents/anastasiia_plan_hazard/agent.py` lines 356-366, `plan_core.py` `_pools`), and `end_chip` 0.6 of
the penalty for every chip at every stage. `cycle_lab` found by hand that the agent starves the small fuel segment
of a grid whose large segment is full, with free routes all episode (49 bn USD in one grid-episode), and named the
cause "in the objective a GWh is a GWh". A fuel is worth something only together with the other fuels of its grid
(`g_av = sum_k min(cap_k x ration_k, I_k)`, base load first), and a fab grid's gate fuel is worth 25 to 40 mn USD a
GWh in a week it completes against 4.1 mn otherwise.

**Formulation.**

- *Audit (the falsifier and the label factory at once).* For sampled states of the model's own play and for each
  stock coordinate j (fuel k at grid g, at its terminal, in a strait's queue, on the way; wafers at source and fab;
  raw chips; packed chips), measure the true marginal value by a paired perturbation: build the state with +delta
  of j (the state is fully visible in the observation and the agent's own simulator copy steps from it, `hub/
  FINDINGS.md`, "Стан симулятора повністю видно"), play both states to the end under the same scenario and the same
  tail policy, subtract. Two step sizes: a small one and "one week of the grid's burn of that fuel", so the
  whole-week kink is measured and not smoothed. Tail policy: the rules for all states (no noise, 0.04 s on Small);
  the planner on a subsample, in aggregates only. Beside it, write the model's own price of the same coordinate:
  the dual of its initial-stock row in the week's exact solve (HiGHS returns `marginals`; not checked that
  `plan_core` keeps them) and, for the window end, the constant.
- *What the audit prints.* By context (grid role: fab grid or not; fuel role: gate fuel or not; own-segment fill;
  the least-covered other segment of the same grid; weeks of cover; weeks left; state of the routes into the node):
  mean true value, mean model price, and their ratio with an interval. A systematic ratio far from 1 in a context is
  a mechanic, with its size in USD.
- *Fit, only where the audit shows a gap.* A concave piecewise-linear credit with the min structure of the
  simulator: for each grid a "complete weeks of cover" variable `w_g <= P_{g,k} / burn_{g,k}` for every fuel k of the
  grid, paid `c_{g,1} >= c_{g,2} >= ...` for the first, second, ... week, plus a small per-unit price for the rest.
  A min of linear functions is concave, so the LP holds it with a handful of columns and rows per grid and no
  integers. The numbers `c` are a function of role-level features (fab margin of the grid, weeks left, whether the
  routes into the grid are cut), fitted by least squares on the audit's values *and* slopes (value and derivative
  from one pair of rollouts), with the model's duals as a prior and one shrink factor. Role-level features, read
  from `config`, make one fit serve Small and Full.
- *Where it plugs in.* The pools of `plan_core._pools` (they already are columns with caps) and the same credit in
  the replay's cost, since the end credit is "part of the cost here and in `simulate` alike". Because it changes the
  judge's accounting too, acceptance is by paired play only.

Data and cost (estimate): 2 000 states x 40 coordinates x 2 steps with the rules tail is under an hour of one core
on Small; Full, with 0.5 to 2 s a tail, about 40 CPU-hours. Planner-tail cross-check of 200 states x 20 coordinates:
20 CPU-hours on Small, cloud on Full.

**Why the information is in the state.** The features are stocks and covers the agent observes, and the label is a
derivative under paired scenarios: the scenario's variance cancels exactly, which is what makes a quantity of order
1e10 visible against an episode of 1.9e12. SPEC's objection to realised-cost labels ("the derivative is seen only as
differences between episodes, under noise a hundred times the signal") is about regression across episodes and does
not apply to a pair on one scenario. Averaged over states of a context, the label is the context's expected marginal
value, which is the causal quantity.

**Why it is not a dead idea.** Not the constant-weight family (CEM out of sample +0.0007; P11's constant stock
premium null): there a noisy episode score selected the number; here the number is measured. Not P5 as a
prohibition (-0.0140, the cell went infeasible): this is a price, concave by construction. Not a new judge: the
credit is validated against the simulator's finite differences before it is used, which is the answer to the P2
reviewer's question whether a V fitted from the model inherits the model's miscalibration. It is the missing ceiling
measurement of P8 ("best constant against best state-dependent weight"), obtained as a by-product.

**Size (estimate).** Small +0.003 to +0.006, Full +0.004 to +0.010. Pools: the non-information difference (0.006 to
0.018), the level-1 residue with the fuel-mix mechanism (30 bn an episode on 4 episodes, information included), the
family's payments so far (`end_left` +0.0025, second round with the bound +0.0034 / +0.0067). Full has 78 capped
weeks of 104 against 26 of 52, five gate-fuel grids and fabs whose chips sell, so the same error costs more there.
If the credit holds, a shorter window becomes possible (P2's gate with H = 10) and frees CPU on Full.

**Cheapest falsifier.** The audit itself on 200 Small states with the rules tail (minutes): if the model's prices
agree with the finite differences in every context within their intervals, there is nothing to fit. One targeted
cell first: the scarce segment of a fab grid whose other segments are at cap, against any segment already at cap.
Ceiling before any fit: play 16 episodes with the *measured* finite-difference prices of each window end (computed
offline with the true future) in place of the constants; under +0.003 on both networks closes the fit.

**Online CPU and overfitting.** A few dozen columns and rows, no solve added. Tens of fitted numbers from thousands
of states of root 1001; stability checked on halves and on root 1002; paired play on root 1003; the four-row gate of
P2 (base; V at H 26; V at H 10; H 10 without V).

## 3. Forked futures: exact expected advantages, a learnability gate, and the price of insurance

**Formulation.** Build a scenario generator that keeps the past of an episode and redraws its future, then use it
for three measurements nobody could make so far.

- *The tool.* Every draw of the generator is keyed (`omega/seeds.py`: immigrants by (block, region, week), offspring
  by the parent's key, durations by the event's key; one generator per regime chain). A fork at week t keeps every
  event that started by t, redraws the residual duration of each running event from its law conditional on its age
  (`disruption/laws.py`; for a carried event the age is the true one, a small leak to note), redraws the regime
  chains and immigrants after t under another entropy, and redraws the offspring of running parents that would land
  after t. The marks of the spliced event list (`marks.graph_marks`) drive the agent's own simulator copy from the
  observed state, so no environment replay is needed. The agent does not read `warning`; its chip part reads
  `pending_prohibitions` (`lp_part.py`, lines 102-104), so the first version carries that list over from the
  original episode (an inconsistency with the redrawn future, to be noted with the numbers). A cruder first version
  needs no generator code: keep the running events with redrawn residuals and add the post-t onsets of another
  episode.
- *Measurement A, the learnability gate.* For 100 states x K = 16 futures x a set of candidate first-week actions,
  with the rules as the tail: split the variance of the advantage into between-state variance of its mean over
  futures (what any model of the state can learn) and within-state variance across futures (what none can). The
  multiplier RL measured only the grand mean. This number says, before any training, how much of a +0.31 % weekly
  advantage is a function of the state.
- *Measurement B, the value of hedging at one decision.* Candidates are the planner's own plans under different
  beliefs (as observed; hazard median; every running cut over within a week; every running cut for the whole window;
  one more week of stock at each gate-fuel terminal). For each state, the candidate with the best *mean* over
  futures against the default. The mean gain over states is the value of the stochastic solution for this family,
  exact up to the tail policy. It has never been measured; what exists is one joint two-forecast program
  (+0.004 to +0.006 Small, -0.004 to +0.001 Full over `plan_hull`), margins (-0.003 to -0.012) and a selection
  between two plans by mean cost under a crude second forecast (-0.012).
- *The learned object, if B is positive.* Not a scenario program online (K scenarios multiply columns and Full has
  no room). The stochastic solution differs from the point-forecast one by holding stock; the same paired
  perturbation as in idea 2, averaged over forked futures, gives the expected marginal value of a week of stock at
  each node given the observable status of the routes behind it (which cuts, of what kind and age). That is a table
  of insurance prices by (route status, weeks of cover), added to the objective as a concave credit.

Cost (estimate): the tool is a day of code. A and B with the rules tail are minutes to an hour on Small plus 600
plans; with the planner tail 40 CPU-hours on Small and about 300 on Full.

**Why the information is in the state.** By construction: the label is a mean over futures that share the state.
What the table conditions on (kind, depth, age of the cuts on a node's routes) is what the tracker already reads
with recall 1.00.

**Why it is not a dead idea.** Hedging by margin died because jumps are rare and large; hedging by an expected
marginal value of stock is the scenario answer the same finding asked for, at the cost of a few objective
coefficients. P11's constant premium on rationed stock was null on Small; a premium that depends on route status is
a different object, and Full was not played.

**Size (estimate).** Small 0 to +0.003, Full +0.002 to +0.006. Pools: onsets (+0.003 / +0.011 with perfect
knowledge, stale), ends of long events (+0.0016 Small), and on Full the measured constant advantage of carrying a
quarter more fuel to TW, KR and CN for the rules agent, with 26 % of gas routes blinking against 10 % on Small.

**Cheapest falsifier.** Measurement B with the crude fork and the rules tail on 100 Small and 50 Full states: a mean
gain of the best-in-expectation candidate under 0.05 % of cost-to-go closes hedging for this planner. Measurement A
closes or opens every per-state learner at once.

**Online CPU and overfitting.** A table lookup. The table is a mean over thousands of forked futures on a root of
its own; the risk is not memorisation but the tail policy (rules against planner), so the confirmation is paired
play.

## 4. The money-no-object causal ladder: decide distillation before training anything

**Formulation.** An offline measurement, then possibly a distillation. Play the same causal agent with every limit
removed, one rung at a time, on fresh episodes of both networks: (a) no clock, search every week (on Full the
search runs in one week of five now); (b) window 39 and 52 weeks; (c) search 60, hull rounds 5, two passes; (d) a
two-stage scenario program with K = 4 to 8 forked futures sharing the first week, fuel block only, as P3 specified.
Cost (estimate): rungs a to c about 20 CPU-hours on Small x64 and 150 on Full x32; rung d about 200 CPU-hours on
Full.

**Why this comes before any distillation.** Distilling an expensive planner into 2 to 4 s presupposes an expensive
causal planner that beats the cheap one. What is measured says there may be none: search 60 against 8 is +0.0004;
`passes` 2, `search` 16, `search_room` 3, `hull_rounds` 2 are all null on both networks (`hub/tried/mpc.md`, P11
block); search every week on Full is +0.0023. The rungs not measured are the long window and the scenario program.

**What to distil, if a rung pays.** Not actions: a model in place of the first LP kept 65 to 68 % of its gain, and
the plan of the expensive agent depends on its own later replans. Distil what the expensive agent knows and the
cheap one lacks, in the cheap one's own language: the value of the state at week 26 as the 52-week agent prices it
(its duals at that week: a label for idea 2's credit), and the stock the scenario program holds against the point
forecast (a label for idea 3's table). Both are prices, both enter as generators.

**Size.** Zero by itself. If rung b or d shows +0.005 or more on a network, it sets a measured target for ideas 2
and 3 and gives them a second, independent label. If the whole ladder is under +0.003, every "expensive planner
into a small model" proposal is closed with one run.

**Falsifier.** It is one. Rung b alone on Small x64 costs about 10 CPU-hours.

**Overfitting.** None in the measurement (fresh root, paired to the base).

## 5. Evidence-driven structure search on cloud machines (P11, widened)

**Formulation.** P11 let an LLM author write price terms through a hook that accepts only a price vector and found
+0.002 on both networks from 20 candidates, blind. Two changes.

- *Three hooks instead of one*: `forecast(observation, tracker) -> window network`, `price(...) -> objective
  vector`, `ask(plan, observation) -> requests`. Both gains of 9 October sit in the first and the third, which the
  present hook cannot express.
- *The author reads evidence, not only code*: the cluster table of idea 1, the audit table of idea 2, the list of
  snapshot-against-week fields. A candidate is a hypothesis about one table cell.
- *Racing on fresh scenarios*: 128 fresh Small episodes a candidate at the first stage (P11 measured that 32 tell
  nothing under +0.003 and a placebo reaches +0.0015), 512 for survivors, Full x64 then x256, confirmation on
  another root under the meter; placebos in every batch set the acceptance threshold. About 500 CPU-hours for 200
  candidates (estimate), four hours of the 128-CPU quota.

**Why the information is in the state.** Each candidate is a causal program; the judge is paired play.

**Why it is not a dead idea.** It is the one learning-type procedure in the repository that already paid on both
networks under a protocol three validators went through. The widening points it at the places where the larger
gains were found.

**Size (estimate).** +0.002 to +0.005 beyond the +0.002 already confirmed, overlapping ideas 1 and 2: it is their
harvesting loop more than a separate pool. Without the evidence tables I would expect the yield of the first run
again, a few terms at the edge of detection.

**Falsifier.** Re-express the early request and the uniform multiplier as candidates of the `ask` hook and check
that the racing protocol recovers them at stage one. If it cannot see +0.003 and +0.001 effects that are known to be
real, it will not find unknown ones.

**Online CPU and overfitting.** Each hook is arithmetic around the solves. Fresh episodes per candidate, placebo
threshold, second root, root 222 untouched.

## 6. More judgements per solve: a proposer of plan perturbations for the replay judge (conditional)

**Formulation.** A candidate costs 0.04 s of cell, 10.78 s of solve and 0.13 s of replay on a Full episode
(`cycle_lab`, E9), so the judge could afford tens of candidates a week if they needed no solve. Candidates are edits
of the solved plan's first weeks (move a week of a scarce segment's fuel one week earlier, swap a tanker release
between routes, hold a wafer order), a learned proposer ranks the edit types by context so that the best five to ten
are tried, the replay picks.

**Why I rank it last and call it a probable trap.** Three measurements point against it: replaying recorded later
actions after a changed first week "loses almost everything, because the value of a change is in the reaction to
it" (`hub/FINDINGS.md`), so the judge needs a reacting tail, which is the rules at about 1 s a variant on Full; the
one family whose ceiling was measured this way is worth +0.0018 on Small and nothing on Full; and a choice between
ready plans by mean window cost measured -0.012.

**Falsifier before any learning.** Offline, no budget: each week try all edits of a fixed list of about 100 with the
existing judge, keep the best, play 16 Small and 8 Full episodes. Under +0.003, closed; a proposer cannot beat the
exhaustive sweep it imitates.

## 7. Traps, said plainly

- **Learning the latent regime or hazard state from the observation history.** Checked in the code and by a count.
  (i) The warning score is S = a X + sqrt(1 - a^2) W, and W is an AR(1) path with the *same* rho as X
  (`disruption/regime.py` lines 514-515). Then S and Q = sqrt(1 - a^2) X - a W are independent AR(1) processes and
  X_t = a S_t + sqrt(1 - a^2) Q_t, so E[X_t | S_1..S_t] = a S_t: the whole history of scores says nothing beyond the
  last score. No filter, recurrent net or smoother can beat the single number already measured at AUROC 0.51 to
  0.59. (ii) Self-excitation: of the onsets of capacity-cutting events inside an episode (sanction, militarised
  closure, conflict; 4.07 an episode), 29 % have a parent that was visible at an earlier observation, the median
  lead from parent to child is 1.2 weeks, and after a visible clustered onset the chance of a cutting onset in the
  same region within three observations is 0.145 against a base of 0.014 (Small, root 333, episodes 2000-2054, one
  run of 45 s). Ten times the base rate, and still one in seven per region, less per edge, with a week of notice,
  against a pool of +0.003 / +0.011 for perfect knowledge of the whole window. (iii) Decoys are thinned per thread
  by one uniform ("a later message on a thread tells nothing about whether it is real", `information/decoys.py`),
  and the ends of events are announced nowhere. The channel was designed so that this is not learnable; the only
  use of onset probabilities is as an input to idea 3's table.
- **A learned ranker for the present proposals.** The family it would rank is worth +0.0018 on Small from a fixed
  state, its cheap neighbourhood on Full is empty to 0.01 % of cost, and three hull rounds with search 8 already
  take most of it. A ranker makes sense only for a richer family, and that family has to be found first (ideas 1, 2).
- **Imitating actions**, of the LP (65 to 68 % kept) or of an oracle (its deviations are the future). **End-to-end
  RL on flows** (no state-dependent advantage on Small for the family tried; idea 3A is the gate for any other
  family). **A per-episode or per-week bandit over planner settings** (the chaos floor of 0.009 an episode is larger
  than any setting's effect; four settings measured null). **Learning to make the week cheaper** (on Small time is
  free, on Full the measured value of a search every week is +0.0023).
- **Fitting a value on realised cost-to-go across episodes.** The repository already says so; the paired
  perturbation of idea 2 is the form that survives.

## 8. Order, cheapest decisive step first

1. `look: 1` oracle on both networks (ceiling of action-week mechanics; code exists).
2. Dual audit on 200 Small states with the rules tail (minutes); the targeted cell first.
3. Diff pass of idea 1 on a fresh root, both networks; read level-1 episodes first.
4. Crude fork and measurements A and B with the rules tail.
5. Ladder rung b (window 39 / 52) on Small x64, then Full on rented machines.
6. Only then fits (idea 2, idea 3) and the widened author loop (idea 5) on the 128-CPU quota.

Steps 1 to 4 need no cloud and no training. None of them may run beside a formal evaluation.

New roots are needed for label generation, table validation and confirmation (for example 1001, 1002, 1003; 777 and
888 are taken by `evolve_lab`); they have to be added to the list of taken roots in `CLAUDE.md` by a person.

## 9. What was run for these notes

One process, 45 s, low priority, root 333 only: `disruption.sampler.sample_events` on Small, episodes 2000-2054,
counting parents of in-episode onsets from the genealogy (`MarkedEvent.parent`). Output:

```
sanction: in-episode onsets/ep 3.35; child of any parent 0.62; parent visible at an earlier observation 0.30
mil_closure: in-episode onsets/ep 0.35; child of any parent 0.63; parent visible at an earlier observation 0.32
conflict: in-episode onsets/ep 0.38; child of any parent 0.48; parent visible at an earlier observation 0.19
all u/o-cutting onsets/ep 4.07 with earlier-visible parent 0.29
lead parent->child, weeks q25/50/75: 0.73 1.19 1.81
P(u/o-cutting onset in same region within next 3 obs | clustered onset seen): 0.145, triggers/ep 8.67
base rate, any region-week: 0.0143
```

"Visible at an earlier observation": the parent's onset is at or after instant 0 and an integer instant lies between
the parent's onset and the child's. Checked by one run on 55 episodes; Full not run. The script was kept outside the
repository (a scratch folder).

Also read, not run: `agents/anastasiia_plan_hazard/agent.py` lines 356-366 and `plan_core.py` `_pools` (the
window-end credit), `disruption/regime.py` lines 512-515 (X and W drawn with one rho), `information/decoys.py`,
`omega/seeds.py` (keyed draws).

Not verified: that `plan_core` keeps the solver's marginals; the list of snapshot fields beyond `u`, `open` and
`kappa`; that a stock perturbation through the agent's simulator copy is as exact as the environment snapshot; every
CPU-hour figure above (estimates from 0.25 to 0.7 s a Small week and about 2.2 s a Full week); every size marked
"estimate".
