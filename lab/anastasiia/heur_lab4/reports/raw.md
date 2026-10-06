# raw: what the LP does differently on fab -> plant slots

Agent: outputs/heur4/agents/raw (chip_part.py changed only; diff: raw.diff). Scratch/code: worktree lab/raw/ (diag.py, an.py, an2.py, wk.py, hyb/).

## Pattern (Small, root 111)
Hybrid restricted by slot group (LP replaces only those raw slots, v2 elsewhere), Small x32, vs v2 0.8100:
- by time: weeks 1-12 +0.0022 (-0.0009..+0.0054), 13-38 +0.0030, 39-52 +0.0033 (gain is spread over the episode)
- by plant: osat_my +0.0072 (+0.0046..+0.0100), osat_tw -0.0007, osat_kr -0.0027 (LP is WORSE into kr/tw)
- by chip: chip_le +0.0078, chip_mat +0.0007
- osat_my x chip_le by source fab: kr_memory +0.0049 (+0.0031..+0.0069), jp_memory +0.0026, tw_leading -0.0002, eu_leading +0.0007
So the gain is about who gets osat_my's room. kr_memory has two plants (osat_kr, osat_my). In v2 its spill into osat_my competes in the same max-flow with jp and eu, whose only plant is osat_my, and takes their room. The LP gives osat_my to the fabs that have nowhere else to go and sends less of kr's chips there (per-slot, 7 episodes: kr->osat_my v2 2.05 M vs LP 1.53 M per episode; jp->osat_my LP 2.6 M vs v2 2.05 M; eu_mature->osat_my LP 3.3 M vs v2 1.7 M).
Other observations (state of the hybrid path, 7 episodes): when v2's room for a plant is 0 the LP still ships about 80 k/week into osat_my (le and mat), 26-44 k into osat_tw (about the plant's outlet); when v2's room is OK it ships about twice what the LP does (osat_my le 211 vs 118 k/week, osat_kr 181 vs 106, osat_tw 123 vs 58). Account lines (lp_raw vs v2, Small x16): shortage 898.9 vs 906.2 bn, tariff 4.0 vs 4.8, disposal 3.2 vs 3.5; raw disposed at fabs falls (kr 0.56->0.09 M, jp 0.46->0.15, eu_mature 0.68->0.28) but 1.06 M packaged chip_le is disposed at plants (v2 0.06 M): disposal is moved, not removed. Imitating that (overfilling or a floor) does not work, see table.

## Change
`PARAMS["raw_captive"]` (default True; False = today's code; "tiers" = one pass per number of reachable plants). `_raw_flows` splits the fabs of a raw chip by the number of plants their alive routes reach; fabs with one plant ("captive") get their max-flow first (`_raw_pass`, the old graph code moved into a method), then the others fill the rooms that are left (room less what the first pass took, backlog plus it). With False or no captive/flexible mix the result is identical to v2 (checked: base_r = 0.8104 identical rows). Also added, default off: raw_floor, raw_cap, raw_arrive, raw_over (experiments below). Functions touched: `_raw_flows`, new `_raw_pass`, `_plant_state` (switches, defaults off).

## Variants (costs.py, root 111, base = v2; interval 90%)
| variant | task | eps | score | diff | interval |
| --- | --- | --- | --- | --- | --- |
| raw_captive (final) | Small | 64 | 0.8143 | +0.0038 | +0.0020 to +0.0059 |
| raw_captive | Full | 32 | 0.8466 | +0.0001 | +0.0000 to +0.0002 |
| raw_captive "tiers" | Full | 32 | 0.8467 | +0.0002 | -0.0010 to +0.0014 |
| raw_floor 1.0 (plant with room 0 still takes 1 week of outlet) | Small | 64 | 0.8111 | +0.0007 | +0.0003 to +0.0010 |
| raw_floor 0.5 | Small | 64 | 0.8105 | +0.0001 | 0.0000 to +0.0002 |
| raw_floor 2 | Small | 64 | 0.7989 | -0.0116 | -0.0144 to -0.0088 |
| raw_floor 4 | Small | 64 | 0.7952 | -0.0152 | -0.0191 to -0.0116 |
| raw_cap 2 (plant inflow <= 2 weeks of outlet) | Small | 64 | 0.8106 | +0.0001 | -0.0001 to +0.0004 |
| raw_cap 1 | Small | 64 | 0.7978 | -0.0126 | -0.0159 to -0.0094 |
| raw_arrive 0 (in-transit cargo counts only if it arrives by packaging time) | Small | 64 | 0.8107 | +0.0003 | +0.0001 to +0.0006 |
| raw_arrive 2 / 4 / 8 | Small | 64 | 0.8106 | +0.0002 | 0.0000 to +0.0004 |
| raw_over 0.25 (storage counted 25% larger) | Small | 64 | 0.8009 | -0.0095 | -0.0120 to -0.0070 |
| raw_over 0.5 | Small | 64 | 0.7969 | -0.0135 | -0.0168 to -0.0102 |
Diagnostic hybrids (Small x32): see Pattern.

Not true / not the cause: queued or stuck cargo does not inflate plants' backlog (osat_my mat "538 k in transit" is a steady 6-week pipeline); smoothing or overfilling inflow loses a lot.

## account.py / checks
Not run on the final agent (only on v2 and lp_raw, files outputs/heur4/raw/acc_*.txt). sbf check, longest week: Small 0.011 s (week 1), median 0.0033 s; Full 0.040 s (week 1), median 0.0134 s. Both pass.

## Next
- Full: the gain does not carry (+0.0001). Full has 7 plants and 14 fabs, most flexible; the captive notion is too coarse there. Read the hybrid by slot group on Full (hybrid_full_111.json exists) to find which plant/fab pairs gain there.
- The remaining ~+0.008 on Small is in osat_my x chip_le from tw/eu/jp and the endgame (last 14 weeks +0.0033); both untested as rules.
- A small LP over the raw slots (the fall-back brief) was not tried.
