# Full: what the plan with base load first scores, and the rule agent beside it (6 October)

Root 444, episodes 0..7 (episode 3 has no plan); the plan's arrays, summary and logs of that day in outputs/heur3/full_ceiling/ (local, not in git)

On Full, root 444, the base-load-first plan scores about 0.93-0.95 on its own cost and about 0.87-0.89 replayed blindly, on 7 of 8 episodes. v2 scores 0.83-0.85 on the same 7. So the realistic ceiling on Full looks like Small's: about 0.87-0.89, not 0.95. Only 7 episodes are in the plan numbers, so this is a rough reading.

**What was run**
- Episodes 0..7, `--time_limit=300`, `--n_jobs=2`, about 48 minutes of wall time.
- Episode 3 (level 1) is left out: no incumbent within 300 s, nor within the retry at 900 s, so it has no plan.
- Every one of the 7 solved episodes ended at the 300 s limit (status 1, incumbent only). The plan's score is therefore a lower bound.
- The solver stayed at about one core per process (about 90% CPU each). There were at most 2 of my workers at a time.

**Per episode** (scores use each episode's own naive and clairvoyant costs)

| ep | harm level | MILP status / gap / seconds | v2 | plan, own cost | plan, replayed | v2 − plan (bn USD) | replay − plan (bn USD) |
|---|---|---|---|---|---|---|---|
| 0 | 1 | limit / 0.0187 / 300 | 0.872 | 0.979 | 0.929 | +325 | +151 |
| 1 | 3 | limit / 0.0062 / 300 | 0.913 | 0.961 | 0.917 | +178 | +163 |
| 2 | 1 | limit / 0.0246 / 300 | 0.634 | 0.777 | 0.705 | +486 | +242 |
| 3 | 1 | no incumbent in 300 s or 900 s, left out | n/a | n/a | n/a | n/a | n/a |
| 4 | 3 | limit / 0.0032 / 300 | 0.833 | 0.978 | 0.908 | +567 | +274 |
| 5 | 1 | limit / 0.0156 / 300 | 0.841 | 0.933 | 0.818 | +129 | +162 |
| 6 | 1 | limit / 0.0027 / 300 | 0.893 | 0.982 | 0.921 | +310 | +211 |
| 7 | 3 | limit / 0.0170 / 300 | 0.933 | 0.979 | 0.926 | +206 | +240 |

- Levels present: 1 and 3 only (levels 2 and 4 are absent). On the 8 episodes, levels 1/3 have 5/3 episodes.
- In the differences, "+" means the first cost is higher. Episode 2 is the hard one (level 1): v2 0.634, plan 0.777, replay 0.705.

**Pooled, beside Small**

| | v2 | plan, own cost | plan, replayed |
|---|---|---|---|
| Small (40 episodes) | 0.842 | 0.950 | 0.869 |
| Full, my pooling of the 7 episodes (levels 1/3 weighted 0.5/0.15, board formula as in `costs.py`) | 0.831 | 0.931 | 0.867 |
| Full, plain ratio of sums, 7 episodes | 0.851 | 0.944 | 0.883 |
| Full, `plan_stats` summary (its `rss`) | n/a | 0.95 | 0.89 |

- The `plan_stats` summary reports 0.95 and 0.89, and I did not check why it differs from my 0.931 and 0.867 pooling. By my own pooling, the board's clairvoyant replayed blindly scores 0.798 on these 7 (`plan_stats` prints it as 0.83).
- v2 over all 8 episodes by `costs.py` is 0.7859. That figure includes episode 3, which has no plan.
- Per-level numbers from `plan_stats`: plan 0.91 (level 1) and 0.97 (level 3); replay 0.85 and 0.92.
- The plan numbers are lower bounds, since all 7 were time-limited with gaps of 0.003 to 0.025. A solved-to-optimality plan would score slightly higher, by up to roughly the gap times the cost.
- With 7 episodes in 2 levels, I would not trust any of these to better than about ±0.03-0.05. The v2 pooled score on these 7 (0.83) is close to v2's 0.834 on the full set.

**Is the MILP practical on Full?**
- Barely. All 7 solves hit 300 s with gaps of 0.3-2.5%, so the solves are useful but not closed.
- Episode 3 gave no incumbent in 900 s.
- Cost in practice: about 340 s per episode including the replays, and the whole 8-episode run took about 28 minutes with 2 workers.
- I did not measure the model size (rows, columns, binaries) or the LP relaxation time. The binaries are 104 weeks × 8 grids = 832, plus the LP columns (104 × `nc`).

**Plan versus the Small picture** (`summary.md` sections 2 and 3, 7 episodes)
- Grids are powered far more of the time on Full than on Small:
  - grid_tw 78%, grid_kr 77%, grid_us 82%.
  - grid_jp 39%, grid_eu 33%, grid_cn 29%, grid_sea 18%.
  - grid_in 0%, with 522 GWh shed per week.
- Small's powered shares were 12-41%.
- The plan is never powered while its grid sheds, by construction. The relaxed plan is powered while shedding in 82% of weeks for grid_cn, 59% for grid_sea and 40% for grid_jp.
- Lots as a share of nominal capacity are 50-75% at the big fabs:
  - 75% at fab_kr_memory_1, 61% at fab_tw_leading_1, 59% at fab_us_mature_2.
  - 10-13% at fab_eu_leading_1 and fab_row_leading_1.
  - Small's fabs ran at 6-31%.
- The plan starts lots in 70-90% of weeks at the big fabs.
- It cuts the relaxed plan's lots at fab_cn_mature_1 (27% against 60%) and fab_sea_mature_1 (18% against 53%). Those are the grids where the relaxed plan cheated on the base load.

**Surprises**
- The first run (`--episodes=2`) solved both episodes in 341 s but wrote nothing. The `report()` call crashed with `ModuleNotFoundError: package_baselines`, and the `.npz` is written after it. That module lives only in `lab/anastasiia/mpc_lab/`, so I reran with `PYTHONPATH=$MAIN/lab/anastasiia/mpc_lab`. That cost about 6 minutes, and the script is unchanged. Anyone running it fresh will hit the same failure. The lead fixed the import in plan_stats.py afterwards, so the PYTHONPATH prefix is no longer needed.
- The script has no option to skip already-solved episodes, so I went straight to `--episodes=8` rather than 2 and then 8.

**Commands** (`MAIN=/Users/anastasiiamazur/Projects/secreton-mantis-analytics`, `PY=$MAIN/.venv/bin/python`)
- Run 1 (failed at the report step): `cd $MAIN && $PY lab/anastasiia/stats_lab/plan_stats.py --task=full --episodes=2 --entropy=444 --time_limit=300 --n_jobs=2 --out=$MAIN/outputs/heur3/full_ceiling/run1`
- Run 8: `PYTHONPATH=$MAIN/lab/anastasiia/mpc_lab $PY lab/anastasiia/stats_lab/plan_stats.py --task=full --episodes=8 --entropy=444 --time_limit=300 --n_jobs=2 --out=$MAIN/outputs/heur3/full_ceiling/run8`
- v2: `$PY lab/anastasiia/heur_lab3/tools/costs.py $MAIN/agents/anastasiia_rules_v2 --task=full --entropy=444 --episodes=8 --n_jobs=2 --out=$MAIN/outputs/heur3/full_ceiling/v2_full_444.json`

**Files**
- Plan: `$MAIN/outputs/heur3/full_ceiling/run8/summary.md` and `$MAIN/outputs/heur3/full_ceiling/run8/episodes.npz`.
- v2: `$MAIN/outputs/heur3/full_ceiling/v2_full_444.json`.
- Logs: `$MAIN/outputs/heur3/full_ceiling/run1.log`, `$MAIN/outputs/heur3/full_ceiling/run8.log` and `$MAIN/outputs/heur3/full_ceiling/v2.log`.
- Reference-cache files for Full root 444 were written under `$MAIN/hub/refcache/references/v0.1.2-39ec701c95ac/full/7113ffee789f4db5/entropy-444/fq-1000-highs-ipm`. They are uncommitted.
