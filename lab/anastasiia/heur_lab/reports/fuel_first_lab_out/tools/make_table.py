"""Markdown table of every variant against pull from scratch/table_all.json (scratch)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "scratch" / "table_all.json").read_text())

# code stage of each snapshot (what the agent file did when the snapshot was taken)
STAGE = {}
for n in "v1_c1 v1_c3 v1_nopulse v2_c1 v2_c3 v2_c6 v2_pulse".split() + [f"b_{x}" for x in "c2 c4 end16 end8 m1 m3 m4 on100 on85 pm0 pm3 ss0 ss6".split()]:
    STAGE[n] = "A single pass"
for n in "p_c1t3 p_c1t2 p_c1t4 p_c1t6 p_c2t4 p_c1t3_np p_c05t4 p_c0t4 p_c1t4_ss2 p_c1t4_ss45 p_c1t4_end16 q_c0t4e16 q_c0t4e20 q_cm5t4e16 q_c0t4e16_m1 q_c0t4e16_on85".split():
    STAGE[n] = "B two passes"
for n in "r0 r_cf r_pm0 r_pm0_cf".split():
    STAGE[n] = "C + crude first option"
for n in "s0 s_bank s_bank5 s_bank99 t_default".split() + [f"u_{x}" for x in "e14 e18 fl5 gc1 gc2 m1 m3 on100 on9 pm15 pm3 qf0 qf1 ss2 ss4 t3 t6".split()] + [f"w_{x}" for x in "qf0 qf0_fl1 qf0_fl100 qf0_fl13 qf0_fl2 qf0_qh0 qf0_qh2 qf0_qh8 qf2".split()] + [f"x_{x}" for x in "a b c d e f".split()]:
    STAGE[n] = "D + crude banking option"
for n in "y0 y_wp10 y_wp10_ws3 y_wp3 y_wp6 y_wp6_ws15 y_ws15 y_ws3".split():
    STAGE[n] = "E + wafer gate option"
for n in "z_lcap z_lcost z_none z_pa z_rev a0 a_pm0 a_pm0_m1 a_pm0_m3 a_pm0_on8 a_pm0_on95 f_final final1".split():
    STAGE[n] = "F + priority, lane-order, prime-arrival options (final file for a0, f_final, final1)"
for n in "g0 g0b g_sf5 g_sf7 g_y8 g_y5 g_y10".split():
    STAGE[n] = "G + share_first / yield_ratio options (defaults = final)"
for n in "abl_nopulse abl_nobank abl_nocrudefirst abl_singlepass abl_nucmax abl_novalve abl_nocapacity abl_notransit".split():
    STAGE[n] = "H final agent with one rule removed"

# defaults of each stage that differ from the final file (so the params column is complete)
STAGE_DEFAULTS = {
    "A single pass": "single pass; defaults cover_weeks 1, queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95, end_weeks 12",
    "B two passes": "two passes (top_weeks 3 unless set); defaults queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95",
    "C + crude first option": "defaults cover 0, top 4, end 16, queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95",
    "D + crude banking option": "defaults cover 0, top 4, end 16, crude_first on, queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95",
    "E + wafer gate option": "defaults of the final file except prime_arrivals off",
    "F + priority, lane-order, prime-arrival options (final file for a0, f_final, final1)": "defaults of the final file except prime_arrivals off (z_*)",
}
NOTES = {
    "abl_nopulse": "no time concentration (valve always moves as the level asks)",
    "abl_nobank": "crude is not banked while the grid is off",
    "abl_nocrudefirst": "lng before crude inside each grid's turn",
    "abl_singlepass": "no top-up pass (orders only up to the essential level)",
    "abl_nucmax": "nuclear fuel: ask for all the lanes can carry, as pull does",
    "abl_novalve": "valve moves everything the terminal holds (no closing-stock rule), no time concentration",
    "abl_nocapacity": "requests ignore the lanes' edges, straits and fleet slack: min(need, source stock)",
    "abl_notransit": "position counts stock on hand only (not cargo in transit or waiting at straits)",
    "g0": "= the final agent", "g0b": "= the final agent (G_bar fallback)",
    "g_sf5": "share_first 0.5: every grid first gets half its essential need, then strict priority",
    "g_sf7": "share_first 0.7", "g_y8": "yield_ratio 0.8: grids whose lanes carry < 80% of the burn are served last",
    "g_y5": "yield_ratio 0.5", "g_y10": "yield_ratio 1.0",
    "v1_c1": "no time concentration (pulse off); order-up-to with priority, terminal valve, capacity-aware lanes",
    "v1_nopulse": "same as v1_c1 (first snapshot)", "v2_pulse": "same as v2_c1 (first snapshot)",
    "a0": "= the final agent", "final1": "= the final agent (clean file)", "f_final": "= the final agent",
    "r0": "= q_c0t4e16", "s0": "= r_cf (crude first on)", "y0": "final defaults but prime_arrivals off",
    "t_default": "crude first + banking on, queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95",
}

lines = ["| variant | code stage | parameter overrides (params.json) | score | diff vs pull | 90% interval of the diff |", "| --- | --- | --- | ---: | ---: | --- |"]
order = sorted(rows, key=lambda n: (list(dict.fromkeys(STAGE.values())).index(STAGE.get(n, "F + priority, lane-order, prime-arrival options (final file for a0, f_final, final1)")), n))
for n in order:
    f = rows[n].split()
    score, diff = f[0], f[1]
    interval = " ".join(f[2:5])
    p = json.loads((ROOT / "scratch" / "variants" / n / "params.json").read_text()) if (ROOT / "scratch" / "variants" / n / "params.json").is_file() else {}
    note = NOTES.get(n, "")
    pj = ", ".join(f"{k}={v}" for k, v in p.items()) or "(none)"
    if note:
        pj += f" -- {note}"
    lines.append(f"| {n} | {STAGE.get(n, '?')} | {pj} | {score} | {diff} | {interval} |")
(ROOT / "scratch" / "table_full.txt").write_text("\n".join(lines) + "\n")
print(len(lines) - 2, "rows")

# ---- compact version: stage codes and a legend ------------------------------------------------------------------
CODES = {
    "A single pass": "A",
    "B two passes": "B",
    "C + crude first option": "C",
    "D + crude banking option": "D",
    "E + wafer gate option": "E",
    "F + priority, lane-order, prime-arrival options (final file for a0, f_final, final1)": "F",
    "G + share_first / yield_ratio options (defaults = final)": "G",
    "H final agent with one rule removed": "H",
    "I + run_band / dynamic end rule options (defaults = final)": "I",
}
for n in "h0 h_rb95 h_rb9 h_rb8 h_rb6 k0 k_d12m2 k_d12m3 k_d12m4 k_d12m6 k_d11m3 k_d13m3 m_rb9_sf5 m_rb9_sf4 m_rb92 m_sf3 final2".split():
    STAGE[n] = "I + run_band / dynamic end rule options (defaults = final)"
NOTES.update(
    {
        "h0": "= the agent before run_band/share_first were adopted",
        "k0": "= the agent before run_band/share_first were adopted",
        "final2": "= THE FINAL AGENT (run_band 0.9, share_first 0.5, final file)",
        "m_rb9_sf5": "= the final agent",
        "m_sf3": "share_first 0.3",
    }
)
compact = [
    "stage legend: A single-pass orders | B two-pass orders | C + crude_first option | D + crude banking | E + wafer gate | "
    "F + priority/lane-order/prime-arrival options | G + share_first/yield | H ablation of the final agent | "
    "I + run_band / dynamic end rule; a0, f_final, final1, g0, g0b, h0, k0 are the final agent",
    "",
    "| variant | st | overrides | score | diff vs pull | 90% interval |",
    "| --- | --- | --- | ---: | ---: | --- |",
]
for n in sorted(rows, key=lambda n: (CODES.get(STAGE.get(n, ""), "Z"), n)):
    f = rows[n].split()
    pf = ROOT / "scratch" / "variants" / n / "params.json"
    p = json.loads(pf.read_text()) if pf.is_file() else {}
    pj = ", ".join(f"{k}={v}" for k, v in p.items()) or "-"
    if NOTES.get(n):
        pj += f" ({NOTES[n]})"
    compact.append(f"| {n} | {CODES.get(STAGE.get(n, ''), '?')} | {pj} | {f[0]} | {f[1]} | {' '.join(f[2:5])} |")
(ROOT / "scratch" / "table_compact.txt").write_text("\n".join(compact) + "\n")
print("compact", len(compact) - 4, "rows")
