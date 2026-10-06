"""Markdown rows for the report from compare.py outputs (lines ending in .../lab_out/variants/<name>)."""
import json
import re
import sys
from pathlib import Path

WT = Path(__file__).resolve().parent
DESC = {
    "a_noq": "rule 1 only (no queue ramp)",
    "a_q1_3": "rule 1 + 2, ramp 1 to 3 weeks",
    "a_q2_6": "rule 1 + 2, ramp 2 to 6 weeks",
    "a_q4_12": "rule 1 + 2, ramp 4 to 12 weeks",
    "a_q6_18": "rule 1 + 2, ramp 6 to 18 weeks",
    "b_off": "rule 1 switched off (the copy of pull)",
    "c_base": "v0: rule 1 + 2, ramp 3 to 8 weeks",
    "c_osat": "v0 + plant balance (rule 3, raw chips into a plant)",
    "c_fab2": "v0 + fab balance, buffer 2 weeks of capacity",
    "c_fabS": "v0 + fab balance, up to its storage",
    "c_both": "plant balance + fab balance (2 weeks)",
    "d_fill": "plant balance, outflow sized with shared edges split (osat_fill)",
    "d_stock": "plant balance + stock rows in the allocation",
    "d_rr1": "plant balance + stock rows + reroute of cut volume",
    "d_fabS": "plant balance + fab balance (storage)",
    "d_last0": "plant balance + last weeks (first version: requests scaled, fuel cut too)",
    "d_last1": "same, one week of margin",
    "d_cover": "plant balance + fuel cover (first version: requests scaled)",
    "e_grp": "plant balance as a row of the allocation (no change)",
    "e_grpFab": "+ fab balance (storage)",
    "e_grpRR": "+ stock rows + reroute",
    "e_grpRRFab": "+ stock rows + reroute + fab balance",
    "e_cover": "plant balance + fuel cover (as a row; still without the lng rationing threshold)",
    "e_lastc": "plant balance + last weeks (chip chain only)",
    "e_weeks6": "plant holds at most 6 weeks of its outflow",
    "e_weeks12": "plant holds at most 12 weeks of its outflow",
    "e_margin2": "market takes 2.0 x forecast in a plant's outflow",
    "e_margin1": "market takes 1.0 x forecast in a plant's outflow",
    "e_split": "market's ask split by what a route can carry (all commodities, fuel too)",
    "g_last": "best so far + last weeks (chips)",
    "g_cover": "best so far + fuel cover (with threshold, horizon wrong)",
    "g_split": "best so far + route-capacity split of a market's ask (chips only)",
    "g_pf": "best so far + priority fill",
    "g_tar02": "best so far + tariff preference, scale 0.02",
    "g_tar10": "best so far + tariff preference, scale 0.10",
    "g_rel": "best so far + release override (rule 6)",
    "g_lastcover": "best so far + last weeks + fuel cover (wrong horizon)",
    "h_cover": "best so far + fuel cover (horizon now to the end)",
    "h_fleet": "best so far + fleet-slack rows",
    "h_pfLast": "best so far + priority fill + last weeks",
    "h_pfLastFleet": "same + fleet-slack rows",
    "h_store2": "best so far, a plant may fill 2 x its storage",
    "h_store4": "best so far, a plant may fill 4 x its storage",
    "i_all": "FINAL: plant + fab balance, stock rows, reroute, priority fill, last weeks (chips), fuel cover",
    "i_store08": "final, plant may fill 0.8 x storage",
    "i_store12": "final, plant may fill 1.2 x storage",
    "i_q412": "final, queue ramp 4 to 12 weeks",
    "i_sink10": "final, market margin 1.0 (pull's 1.3)",
    "i_sink16": "final, market margin 1.6",
    "i_fab4": "final, fab buffer 4 weeks of capacity (instead of storage)",
    "j_fillp": "final, plant outflow sized with the commodity order on shared edges",
    "k_blocked": "final, release override only where a queued lane is blocked",
    "l_ontemplate": "rules on top of send-the-maximum (pull's two rules off), no priority fill",
    "l_ontemplate_pf": "rules on top of send-the-maximum (market margin None), priority fill",
    "n2_w1": "final, raw chips weighted by their plant's outflow (power 1)",
    "n2_w2": "final, raw chips weighted by their plant's outflow (power 2)",
    "m_noOsat": "final without plant balance",
    "m_noFab": "final without fab balance",
    "m_noRR": "final without stock rows, reroute and priority fill",
    "m_noPF": "final without priority fill",
    "m_noLast": "final without last weeks",
    "m_noCover": "final without fuel cover",
    "m_noQueue": "final without the queue ramp",
    "m_noLane": "final without the allocation (rows only scale requests): not a clean ablation",
}

rows = []
for f in sys.argv[1:]:
    for ln in Path(f).read_text().splitlines():
        m = re.match(r"\s*(-?[0-9.]+)\s+([+-][0-9.]+)\s+([+-][0-9.]+) to ([+-][0-9.]+)\s+(\S+ \S+ \S+ \S+)\s+(\d+)\s+(\S+)$", ln)
        if not m:
            continue
        score, diff, lo, hi, levels, naive, path = m.groups()
        name = Path(path).name
        if "variants" not in path:
            continue
        rows.append((name, DESC.get(name, ""), score, diff, lo, hi, levels))
print("| variant | what | score | vs pull | 90% interval of the difference | harm level 1..4 |")
print("| --- | --- | --- | --- | --- | --- |")
for name, desc, score, diff, lo, hi, levels in rows:
    print(f"| {name} | {desc} | {score} | {diff} | {lo} to {hi} | {levels} |")
