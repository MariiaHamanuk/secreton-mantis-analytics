@AGENTS.md
Single source of truth for how we measure, what we already know, and what we do. If something here contradicts your assumptions, update this file first, then work.

Goal
Build agents/<name>/agent.py (Agent(config) once per episode, act(obs) once per week) that minimises supply-network cost. Metric: RSS = (naive − yours) / (naive − clairvoyant), weighted over harm levels with p = (0.50, 0.30, 0.15, 0.05). Negative RSS is kept.

Dev (public board): Small-v0, 200 hidden episodes, until 10 Oct 23:59 (Kyiv).
Final: Full-v0, 400 hidden episodes, 11 Oct 00:00–13:00. The submission that sits on the board is the one re-run.
CPU per week: 2 s (Small), 4 s (Full). A week over budget is played by the naive rule (fallback).
Submission limits: 3 per UTC day, agent.py at zip root, Python 3.13, NumPy, SciPy, PyTorch (CPU, 1 thread), no network.
What we already know (findings)

Status: [verified] confirmed by two independent runs, [1 person] one person so far, awaiting a second check.

Gamma (shock intensity). Gamma 0.62 gives the public split: 20 episodes at entropy 0, harm-level mix 50/30/15/5. Higher gammas exist and can be used to sample more hard cases. [1 person]
We do not know which gamma Full uses. This is the most important thing to find out (see "Open questions").
Therefore final candidates are always evaluated per level 1–4 separately. A heuristic or model can be strong on level 1 and near-naive on level 4. If level 4 turns out to be more than 5% of Full, that agent loses.
20 episodes is a very noisy estimate. Never compare internal variants on it. For control checks on Small use at least 256 episodes, one seed, one gamma. We have already seen RSS differ across seeds. A noise study at 64/128/256+ episodes on Small and Full is in progress. [1 person]
Principle: we need a stable solution that does not swing up or down on subsamples. Stability matters more than a good mean on one sample.
Built-in MPC in the repo. RSS ≈ 0.65 on ≈64 episodes, without forecasting: it plans assuming the whole horizon is a copy of the current state. [1 person]
We are testing the ceiling: the same MPC given the real future ("oracle") at different horizon lengths in weeks. A real forecast of that accuracy is impossible; this is only an upper bound.
@KwenLu reruns this independently (own run, ideally a separate agent), because the result drives our strategy.
Problem: the forecast is probabilistic, LP is deterministic. An LP solver minimises cost in a single world and cannot handle uncertainty. We need both a forecast and a way to use it in the optimisation. Two hypotheses: [hypotheses, untested]
A. Several scenarios in one optimisation, minimising the probability-weighted average cost.
B. Heuristic rules on top of the LP output.
CPU budget: the built-in MPC takes 3.2 s/week on Small against a 2 s limit. [1 person] So as-is it exceeds the budget and some weeks would fall back to naive. Full has a 4 s limit and a bigger network.
The repo ships a Docker build that imitates the server environment. All timing is measured in it. @KwenLu re-measures independently.
Local benchmark
Episode sets (fixed once, never changed)
Set	Purpose	Size
dev20	public root 0, gamma 0.62, entropy 0. Smoke only (does it crash, does it fit CPU)	20
tune	tuning, policy search, training	≥256 on Small, one seed, one gamma
holdout	final check of a candidate; never tune on it	≥256 on Small, different seed from tune
hard	higher gamma if needed, to get enough level 3–4 episodes	separate, report levels separately
Same for Full; size set after the noise study (finding 2).
Episode lists (task, root/seed, gamma, index, level) live in bench/episodes_<task>_<set>.json and are committed to git.
Agents are compared paired only: same episodes, same policy_seed.
256 episodes gives only ~13 level-4 episodes, so level 4 is judged on hard or separately, and never from noise.
Fixing randomness
Seed everything only from config["policy_seed"]: random, numpy (default_rng), torch.manual_seed.
torch.set_num_threads(1), PYTHONHASHSEED=0 in the run script.
No time, os.urandom, uuid or unstable ordering in decision logic.
Training seeds never overlap with tune and holdout.
Determinism test: two runs of the same agent on dev20 give bit-identical costs.
Script

bench/run.py --task {small,full} --set {dev20,tune,holdout,hard} --agent <name>

Reads episodes from the json, plays them as the scorer does (one Agent per episode).
Parallel across episodes (processes), one thread per process.
Writes bench/results/<agent>_<task>_<set>_<githash>.json.
Cross-checks against sbf evaluate on dev20: numbers must match. Exact flags are in the kit README and sbf --help.
What we report
RSS (weighted 0.5/0.3/0.15/0.05) and separately per level 1–4. Mean costs: agent, naive, clairvoyant.
Bootstrap 95% CI for RSS and for the difference between agents (resampling within levels).
Stability: RSS on subsamples (e.g. 4 × 64 from tune). If subsamples disagree by more than the CI, the solution is unstable.
CPU per week: mean, p95, max, measured in the repo's Docker. Number of fallback weeks (must be 0).
CPU budget
Measure with time.process_time(), all threads.
Target: p95 ≤ 50% of the limit (≤ 1 s Small, ≤ 2 s Full), since the server may be slower.
The agent has an internal timer and a fast fallback mode if the main algorithm will not fit. Do not rely on the server's fallback.
Full is mandatory because the Final runs there. A solution fast only on Small does not count.
Load weights and heavy imports at module level, but remember Agent(config) counts toward week 1.
Baselines
naive (RSS = 0)
agents/template
built-in MPC (RSS ≈ 0.65, but 3.2 s on Small, so over budget)
our current best

A candidate goes to submission only if: it beats the current best in a paired comparison on tune (difference CI > 0), it is stable on subsamples, it is no worse on levels 3–4 on hard, it fits CPU on Small and Full, and it has been checked once on holdout.

Directions
First target: an MPC that fits 2 s (Small) and 4 s (Full) without losing RSS. Simplify the model, shorten the horizon, warm start, shrink the LP.
Upper bound: oracle-MPC across horizon lengths (finding 3).
Uncertainty: scenario LP (A) vs rules on top of LP (B). Compare paired on tune, accounting for CPU: several scenarios multiply the problem size.
Forecast: use warnings and announcements (warning.*, messages.* in docs/fields). Some are fake, so tune trust weights.
Open questions
Which gamma does Full use (and does it equal 0.62)? Until answered, evaluate per level and optimise the worst case.
What does "entropy 0" mean for the splits (a seed parameter or a separate knob)? Record the value in the episode json.
How many episodes are needed so RSS noise is smaller than the difference we want to detect (noise study result)?
Will @KwenLu confirm RSS ≈ 0.65, the oracle ceiling and 3.2 s?
Process
Change the agent.
dev20: smoke, no crashes, CPU fine.
tune (Small): paired comparison with the best, CI, stability.
tune (Full): CPU and RSS.
hard: levels 3–4.
holdout on Small and Full for the candidate.
sbf check <name> --task=small, then sbf upload <name> --wait.
Keep on the board only the one we want in the Final ("Add to Leaderboard").

Do not
Do not evaluate variants on 20 episodes, and do not tune on holdout.
Do not draw conclusions without a CI and a subsample check.
Do not change the fixed sets once comparisons have started.
Do not measure CPU outside the repo's Docker, and do not use wall-clock as CPU.
Treat a key result (oracle ceiling, CPU timings) as established only after an independent second check.
Do not burn the 3 daily submissions without a passed holdout.
Always log results with the git hash.
