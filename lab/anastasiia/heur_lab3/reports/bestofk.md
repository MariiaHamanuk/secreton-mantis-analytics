# The best variant of the rule agent per episode, chosen with hindsight (6 October)

Root 111: Small first 64 episodes, Full first 32; variant folders, costs per episode and scripts of that day in outputs/heur3/bestofk/ (local, not in git)

**Result: per-episode selection among v2 parameter variants has an upper bound of +0.006 on Small and +0.008 on Full, so the idea is nearly dead.** The variants differ little from v2, and no week-1 feature predicts which one wins.

Setup: 18 variants on Small (v01 to v18, 64 episodes, root 111). v19 (`nuc_early` off, matters on Full only) also ran on Small, where it is identical to base as expected. 11 on Full (32 episodes, root 111): the 10 that mattered most on Small, plus v19. All 4-episode checks showed costs differing from base, with two exceptions. v19 is identical on Small, as expected. My first `share_first` 1.0 variant was identical to 0.0 (`share_first` >= 1 skips stage 1 and is strict priority too), so I replaced it with 0.8. Base is `anastasiia_rules_v2` (0.8104 Small, 0.8465 Full). "wins" = episodes where the variant beats base by more than 1 cent. "best2" = best of (base, variant) per episode.

## 1. Variants (paired difference vs base, 90% interval)

| Variant | One change | Small score | Small diff [interval] | Small wins | Small best2 | Full score | Full diff [interval] | Full wins | Full best2 |
|---|---|---|---|---|---|---|---|---|---|
| v01_top4 | `top_weeks` 4 | .8098 | -.0006 [-.0012,-.0001] | 35 | .8107 | .8406 | -.0059 [-.0084,-.0034] | 12 | .8471 |
| v02_top12 | `top_weeks` 12 | .8093 | -.0011 [-.0021,-.0003] | 10 | .8104 | not run | | | |
| v03_gbuf04 | `grid_buffer` 0.4 | .8102 | -.0002 [-.0009,+.0004] | 28 | .8111 | .8485 | +.0020 [+.0009,+.0033] | 23 | .8488 |
| v04_gbufoff | `grid_buffer` None | .8083 | -.0021 [-.0035,-.0010] | 10 | .8105 | not run | | | |
| v05_pulseoff | `pulse` False (gas concentration off) | .7108 | -.0996 [-.1181,-.0814] | 8 | .8112 | .7526 | -.0939 [-.1109,-.0774] | 5 | .8471 |
| v06_prime4 | `prime_weeks` 4 | .8051 | -.0053 [-.0077,-.0030] | 17 | .8119 | .8416 | -.0049 [-.0069,-.0033] | 5 | .8468 |
| v07_end8 | `end_weeks` 8 | .8035 | -.0069 [-.0089,-.0050] | 6 | .8105 | not run | | | |
| v08_gateoff | `gate` False (whole-week crude gate off) | .8018 | -.0087 [-.0153,-.0030] | 27 | .8114 | .8314 | -.0151 [-.0256,-.0063] | 9 | .8509 |
| v09_throttleoff | `throttle` False | .7992 | -.0112 [-.0138,-.0088] | 10 | .8109 | .8442 | -.0023 [-.0034,-.0011] | 4 | .8471 |
| v10_share0 | `share_first` 0 (strict priority of grids) | .8088 | -.0016 [-.0032,-.0001] | 23 | .8116 | .8440 | -.0026 [-.0046,-.0008] | 10 | .8473 |
| v11_share08 | `share_first` 0.8 | .8093 | -.0011 [-.0026,+.0003] | 24 | .8116 | .8432 | -.0033 [-.0057,-.0012] | 10 | .8470 |
| v12_fleet07 | `fleet_frac` 0.7 | .8000 | -.0104 [-.0150,-.0063] | 2 | .8104 | not run | | | |
| v13_w2 | `w_weeks` 2 | .8096 | -.0008 [-.0022,+.0005] | 37 | .8119 | .8463 | -.0002 [-.0019,+.0013] | 19 | .8480 |
| v14_w4 | `w_weeks` 4 | .8102 | -.0002 [-.0003,-.0000] | 8 | .8105 | not run | | | |
| v15_rate10 | `rate_cap` 1.0 | .8113 | +.0009 [-.0007,+.0024] | 46 | .8133 | .8489 | +.0024 [+.0013,+.0036] | 22 | .8493 |
| v16_reserveoff | `reserve_dear` and `reserve_useful` False | .8094 | -.0010 [-.0018,-.0002] | 11 | .8109 | not run | | | |
| v17_cover3 | `pack_cover` 3 | .8089 | -.0015 [-.0023,-.0008] | 16 | .8105 | not run | | | |
| v18_ss1 | `ss_weeks` 1.0 (cushion over gas threshold) | .8092 | -.0012 [-.0021,-.0003] | 14 | .8112 | not run | | | |
| v19_nucoff | `nuc_early` False | .8104 | 0 | 0 | .8104 | .8163 | -.0303 [-.0517,-.0110] | 23 | .8485 |

Full levels: [15, 5, 11, 1] (only one level-4 episode). Small levels: [25, 25, 8, 6].

Not varied: the order of grids (no key exists; `share_first` 0 and 0.8 stand in for it).

## 2. Hindsight best-of-K, greedy curve, split-half

| | Small (K=19 incl. v19) | Full (K=11) |
|---|---|---|
| Base | 0.8104 | 0.8465 |
| Hindsight best of all | 0.8168 (**+0.0064**) | 0.8543 (**+0.0078**) |

Greedy curve (score after adding the best next variant):
- **Small:** 0.8113 (rate10), 0.8141 (+share0), 0.8150 (+prime4), 0.8157 (+pulseoff), 0.8161 (+w2), 0.8164 (+gateoff), 0.8165 (+top4), then flat at 0.8168 from about 11 variants.
- **Full:** 0.8489 (rate10), 0.8525 (+gateoff), 0.8534 (+nucoff), 0.8538 (+share0), 0.8540 (+gbuf04), 0.8541, 0.8542, 0.8543 (after 8), then flat.

Split-half: the subset is chosen on one half of the episodes (even or odd index), then the hindsight best-of-m is scored on the other half. Each figure is a test score on the other half. Brackets are the base on the same test halves.
- **Small:**
  - best-of-1: 0.7820 / 0.8354 (base 0.7835 / 0.8366), so the chosen variant is worse than base.
  - best-of-2: 0.7871 / 0.8386, gain +0.0036 / +0.0020.
  - best-of-3: 0.7876 / 0.8387, gain +0.0041 / +0.0021.
  - Mean honest gain at 3 variants is about +0.003.
- **Full:**
  - best-of-1 (rate10 both times): 0.8786 / 0.8290 (base 0.8751 / 0.8275), gain +0.0035 / +0.0015.
  - best-of-2: 0.8802 / 0.8295, gain +0.0051 / +0.0020.
  - best-of-3: 0.8809 / 0.8305, gain +0.0058 / +0.0030.
  - Mean honest gain at 3 variants is about +0.0044. About +0.0025 of that is just `rate_cap` 1.0, a known single-variant gain.
  - The Full numbers rest on 11 variants and 32 episodes.

## 3. Where the gain sits
- **Small:** 30 of 64 episodes give 80% of the gain. Gain share by level 1 to 4: 0.37, 0.41, 0.09, 0.13. As a share of room per level: 0.65%, 0.69%, 0.41%, 1.04%. The biggest single-episode gains are 2 to 3% of room.
- **Full:** 16 of 32 episodes give 80%. Gain share by level: 0.65, 0.11, 0.23, 0.015, so mostly level 1. Level 1 gains 1.1% of room.
- **Which variants win:**
  - **Small:** `rate_cap` 1.0 is the top winner (22 episodes, 35% of the gain). Next are prime4 (17%), pulseoff (14%, only 4 episodes), w2 (10%) and share08 (6%).
  - **Full:** `gate` off takes 46% of the gain from 5 episodes (up to +4.2% of room in one episode, but -12% in the worst). Then rate10 (22%), nucoff (10%) and gbuf04 (7%).
  - The sets overlap only partly: rate10 leads on both boards, but gateoff is the top Full winner on large per-episode swings and a mid-size Small winner.
- **Real per-episode swings** exist only for the structurally different switches (pulseoff, gateoff, nucoff). Their worst-case loss is huge (-0.34, -0.13 and -0.54 of room) against a best case of +0.01 to +0.04. The mild parameters (`top_weeks`, `grid_buffer`, `w_weeks`, `share_first`) swing by about 0.5% of room, which is noise-level.
- **The sum is dominated by many small wins**, not a few large ones. Of the 19 variants, 14 win somewhere on Small, but most winners beat base by fractions of a percent of room.

## 4. No-foresight selector
Two week-1 features: the count of `action_mask`==0 slots, and mean supply availability / max (a crude proxy; the other supply feature was junk). A threshold selector was fit on one half and tested on the other, for rate10, gateoff and gbuf04.
- Spearman correlation of the per-episode difference (diff/room) with the features is weak everywhere. Nominal p-values of 0.04 to 0.08 appear for a few variants out of about 17 tested (for example masked vs top4 on Small rho +0.24, supply vs rate10 on Full rho +0.36), which is what chance gives over this many tests.
- The threshold selectors never beat the better of "always the variant" or "always base" on a test half. For rate10 on Small the halves give +0.0019 / -0.0012. On Full, gateoff selectors give 0 / -0.0044 or -0.0077. Masked count correlates with harm level (rho +0.22 Small, +0.57 Full).
- Nothing obvious separates the winners, so I stopped there.

## 5. Upper bound
Per-episode selection among v2 parameter variants is worth at most +0.006 (0.8168) on Small and +0.008 (0.8543) on Full, against the roughly +0.005 to +0.014 a single-variant swap gives. Honest split-half estimates for a best-of-3 are about +0.003 on Small and +0.004 on Full, mostly `rate_cap` 1.0 plus small extras. Firmness:
- Hindsight best-of-K is optimistic.
- The 90% interval on the hindsight gain over base was about +0.006 to +0.009 on Small; I did not compute it on Full.
- Only 11 variants were run on Full, 32 episodes, one level-4 episode.
- All variants are v2 parameter changes, so a different mechanism or different rules per episode could do more.

A selector with perfect foresight, but allowed only these variants, would gain under 1 point of RSS. A real forward-playing selector would capture less than that. I would not make this the team's next method.

## 6. Files and surprises
All under `/Users/anastasiiamazur/Projects/secreton-mantis-analytics/outputs/heur3/bestofk/`:
- Data: `small_111.json`, `full_111.json`, `small.log`, `full.log`, `feat_small.json`, `feat_full.json`
- Outputs: `small_analysis.txt`, `full_analysis.txt`, `selector_out.txt`
- Scripts: `mk.py` (builds variant folders), `run.sh` (wraps `costs.py`), `analyse.py` (greedy, split-half, where the gain sits), `feat.py` (week-1 features), `selector.py`
- Variants: `agents/v01..v19`

Surprises:
- **`nuc_early` off on Full.** The average is -0.030, but it wins 23 of 32 episodes by small amounts and loses up to 54% of room in a few.
- **Small `gate` off.** Per-episode swing is large, and the average is -0.009 on Small and -0.015 on Full. On Full it is the single largest source of hindsight gain, with 5 winning episodes.
- **Early-warning cost.** The four-episode check cost under 30 seconds, and a Small x64 run took about 35 s per agent, much less than the 1 minute expected.
