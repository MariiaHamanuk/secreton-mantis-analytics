# chip_route: routing the chip chain

Written by the lead from the subagent's hand-back (the subagent could not write files in `MAIN`); its raw records are
in `chip_route_lab_out/records/` (ablations on 64 episodes, `diag.py` and `balance.py` output for pull, the final agent
and mpc) and its scripts in `chip_route_lab_out/tools/`. Agent: `outputs/heur_lab/agents/chip_route/agent.py`. Numbers
are the subagent's. Fuel slots are `pull`'s, entry for entry.

## Result (Small, root 111, 64 episodes, paired against `pull`)

| agent | score | difference | 90% interval | by harm level |
| --- | --- | --- | --- | --- |
| pull | 0.4968 | | | 0.388, 0.578, 0.624, 0.612 |
| chip_route | 0.5679 | +0.0711 | +0.0600 to +0.0826 | 0.464, 0.649, 0.690, 0.661 |

Parameters were tuned on the first 32 episodes of the same root. `sbf check` passes on Small (worst week 16 ms); on
Full the mean act is 7.6 ms. The allocations use a small max-flow (Dinic, graphs under 40 nodes), no LP solver.

## What it does

`Agent.fill_chip_flows(observation, flows)` overwrites the wafer, raw-chip and packaged-chip entries only.

1. **Raw chips, fab to plant.** A max-flow from the fabs' raw stock into the plants, each plant with a room: its
   storage less the backlog its outlets cannot drain while the new chips travel and are packaged. A plant's outlet is
   the capacity of its open routes to markets that want the chip; a plant with no outlet takes nothing, and what no
   plant can take waits at the fab.
2. **Wafers.** Target stock at a fab: 3 weeks of its effective capacity (not above 90% of its storage), times the
   share of its capacity whose chips can still reach a market. Nothing after the week a lot could no longer be sold
   (week 37 or 39 on Small).
3. **Packaged chips, plant to market.** A market's need is 1.3 times its forecast until the cargo arrives, less stock
   and cargo on the way (cargo queued at a strait counted at the week it will really be released). A max-flow fills
   needs in three rounds, chip_le first, faster routes first; chip_mat gets what chip_le leaves of the shared edges.

## What mattered (one rule off, 64 episodes)

| off | change | 90% interval |
| --- | --- | --- |
| raw-chip rule | -0.0502 | -0.0612 to -0.0400 |
| wafer stock 1 week instead of 3 | -0.0151 | -0.0224 to -0.0089 |
| wafer rule | -0.0123 | -0.0156 to -0.0094 |
| packaged-chip rule | -0.0093 | -0.0121 to -0.0071 |
| fill markets to cover | -0.0029 | -0.0041 to -0.0019 |
| plant wait rounds | -0.0027 | -0.0040 to -0.0016 |
| queue-aware arrival weeks | -0.0018 | -0.0039 to -0.0003 |

## Did not help or hurt

Sizing the wafer stock from recent starts (-0.015); gating strait lanes on `graph_now.open` (-0.002 to -0.004:
`kappa` already includes `open`); starving the lower-value fab of a shared grid of wafers (0 to -0.006); any plant
fill other than exactly its storage (-0.004 to -0.015); softer end-of-horizon rules (-0.009 to -0.012).

## Physical effect (8 episodes: pull, chip_route, mpc)

- chip_le served per episode: 8.39, 9.20, 9.65 M; chip_mat: 3.31, 3.97, 4.28 M.
- chip_le disposed of at plants: 1.69 M under pull, 0.04 M; the loss moved to the fabs (0.39 to 1.10 M), where chips
  wait for an outlet instead of being shipped into a full plant.
- Wafers waiting in strait queues: 2,198 to 168 thousand; wafer tariffs 2.2 to 0.4 bn USD per episode.

## Where the rest of the gap to mpc is

Shortage against mpc on episodes 0 to 5: -4, +156, 0, +52, +8, -13 bn USD. The two episodes that carry the gap are
those where mpc starts about twice as many chip_le lots: fuel, not routing.

## Notes for the lab

- Tiny has no `layout["lot_keys"]`; an agent that reads it crashes there.
- The agent's Tiny check wrote 24 reference files under `team/refcache/.../tiny/.../entropy-111/` (episodes of Tiny,
  root 111). They are deterministic caches.
- `Agent.chip_value(observation)` gives, per grid, the USD per GWh and GWh a week of each fab whose chips can still be
  sold: an input for a fuel rule.
