# Heuristics lab 4: three hypotheses worked through, and the hybrid beside them (6 October, evening)

The fourth round on the rule agent. Lab 3 (`../heur_lab3/`) found where `agents/anastasiia_rules_v2` loses and that
its parameters are used up; this round tried the hypotheses that came out of it. The numbers are in
`hub/tried/heuristics.md` (section "Четверта хвиля").

## How the work was done

Three Sonnet subagents, each in its own git worktree, each changing one function of the agent behind a switch; the
lead merged, measured the sum, and ran the hybrid of `../hybrid_lab/` beside it.

| name | hypothesis | what came out |
| --- | --- | --- |
| `oil` | on a first edge shared by several grids, a grid that has enough takes crude from one that is short | wrong: the shared edges are 57 to 66% full and the crude stands at the short grids' own terminals. The follow-up, the terminal-to-grid valve, gave `gate_keep` |
| `cap` | cut a fab's wafers only by what it is seen throwing away | `cap`: the overflow is recovered exactly from the observation; a small gain on Full, none on Small |
| `raw` | find what the linear program does differently on the fab -> plant slots and copy it | `raw_captive`: a third of the program's gain on Small, nothing on Full |

## What is here

- `agents/rules_v3/`: `agents/anastasiia_rules_v2` with the three rules, each a switch of `PARAMS` / `FUEL`:
  `gate_keep` 3.0 (`fuel_part.py`), `cap` and `raw_captive` (`chip_part.py`). No file of its own beyond the four of
  v2; `sbf check` passes on Small and Full (longest week 0.011 s and 0.039 s, outside the container).
- `reports/oil.md`, `reports/cap.md`, `reports/raw.md`: the subagents' reports, every variant with its paired
  difference. Their paths under `outputs/heur4/` and `.claude/worktrees/` are of that day only.

## The numbers (root 111, paired with v2)

| agent | Small | Full |
| --- | --- | --- |
| `gate_keep` 3.0 | x64: +0.0009 (+0.0003 to +0.0014) | x64: +0.0050 (+0.0026 to +0.0080) |
| `cap` | x64: +0.0003 (-0.0002 to +0.0009) | x32: +0.0026 (+0.0014 to +0.0038) |
| `raw_captive` | x64: +0.0038 (+0.0020 to +0.0059) | x32: +0.0001 (+0.0000 to +0.0002) |
| all three, `agents/rules_v3` | x256: 0.8282, +0.0053 (+0.0043 to +0.0064) | x64: 0.8428, +0.0070 (+0.0045 to +0.0101) |

`rules_v3` is better at every harm level on both networks and on every prefix (Small 64 / 128 / 256: +0.0046 /
+0.0058 / +0.0053). The same four files are the model `agents/anastasiia_rules_v3` of `hub/FORMAL_RESULTS.md`:
0.831 on Small and 0.840 on Full (root 222), +0.0046 and +0.0061 over v2.

The hybrid beside it: `../hybrid_lab/agents/hybrid_chiplp` with its `chip_part.py` and `fuel_part.py` replaced by
those of `agents/rules_v3` (the rules decide everything, then the raw-chip and packaged-chip entries come from week 1
of a linear program built by a copy of the package's modules).

| agent | Small x256 | Full x64 | container, Small | container, Full |
| --- | --- | --- | --- | --- |
| v2 | 0.8230 | 0.8359 | 0.01 / 0.04 s | 0.03 / 0.17 s |
| `rules_v3` | 0.8282 | 0.8428 | 0.008 / 0.037 s | 0.031 / 0.104 s |
| the hybrid on `rules_v3` | 0.8384, +0.0154 (+0.0139 to +0.0171) | 0.8553, +0.0194 (+0.0163 to +0.0229) | 0.17 / 0.34 s | 0.92 / 1.41 s |

Seconds are the median and the longest week of one dev episode in the scoring container on a free machine. The
program has its own timer (0.4 of the week's budget) and hands the week to the rules past it.

That hybrid is the model `agents/anastasiia_hybrid_chiplp` of `hub/FORMAL_RESULTS.md`: 0.840 on Small and 0.854 on
Full (root 222), +0.0138 and +0.0193 over v2, and 0.839 on the public board.

## What the round taught

- Fuel thrown away at terminals is mostly harmless, as the team's notes say: of 250 thousand units of lng disposed of
  per Full episode, about 4 thousand could have been burned by the same grid within 16 weeks. What does cost is a
  gated fuel standing at a terminal for the whole episode.
- The gain of the linear program on raw chips is about WHO gets a plant's room, not how much is shipped: every
  variant that ships more (a floor, a larger store, a cap on inflow) loses.
- On Full the program's gain is on both stages (fab -> plant +0.0086, with plant -> market +0.0158) and no rule of
  this round reproduces it.
