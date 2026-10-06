"""Every variant against pull on 64 episodes of root 111, grouped by the stage of the code it was run on (scratch).

Reads scratch/table_all.json (the screen.py line of each variant) and each variant's params.json; writes
lab_out/records/table_variants_by_stage.txt and prints it. Rows whose line and parameters are identical are merged.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "scratch" / "table_all.json").read_text())

STAGES = [
    ("A", "single-pass order-up-to; level = base + burn x (lead + cover_weeks); defaults of the stage: pulse on, cover 1, ss 0.3, "
          "queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95, prime_weeks 2, end_weeks 12, pulse_min 0.1. v1_* have pulse off. "
          "b_* change one number of v2_c3 (cover 3).",
     "v1_c1 v1_nopulse v1_c3 v2_c1 v2_pulse v2_c3 v2_c6 b_c2 b_c4 b_m1 b_m3 b_m4 b_on85 b_on100 b_pm0 b_pm3 b_end8 b_end16 b_ss0 b_ss6"),
    ("B", "two passes (essential level, then a top-up level from what is left); defaults: cover 1, top 3, the rest as A. "
          "q_* have end_weeks 16.",
     "p_c1t3 p_c1t2 p_c1t4 p_c1t6 p_c2t4 p_c1t3_np p_c05t4 p_c0t4 p_c1t4_ss2 p_c1t4_ss45 p_c1t4_end16 q_c0t4e16 q_c0t4e20 q_cm5t4e16 "
    "q_c0t4e16_m1 q_c0t4e16_on85"),
    ("C", "+ crude_first option (default off); defaults: cover 0, top 4, end 16, queue_frac 0.5, fleet_frac 0.85, on_ratio 0.95",
     "r0 r_cf r_pm0 r_pm0_cf"),
    ("D", "+ bank_crude option (default off); s0 = crude_first on; t_default = crude_first and bank_crude on (bank_ratio 0.9). "
          "u_* change one number of t_default; w_* and x_* change queue_frac, fleet_frac and others (listed)",
     "s0 s_bank s_bank5 s_bank99 t_default u_pm15 u_pm3 u_m1 u_m3 u_on9 u_on100 u_e14 u_e18 u_ss2 u_ss4 u_t3 u_t6 u_gc1 u_gc2 u_qf0 u_qf1 "
     "u_fl5 w_qf0 w_qf2 w_qf0_fl1 w_qf0_fl13 w_qf0_fl2 w_qf0_fl100 w_qf0_qh0 w_qf0_qh2 w_qf0_qh8 x_a x_b x_c x_d x_e x_f"),
    ("E", "+ wafer-gate options (w_prime, w_stop); y0 = defaults queue_frac 0.15, fleet_frac 1.0, on_ratio 0.9, crude_first, bank_crude, "
          "cover 0, top 4, end 16 and prime_arrivals OFF",
     "y0 y_wp3 y_wp6 y_wp10 y_ws15 y_ws3 y_wp6_ws15 y_wp10_ws3"),
    ("F", "+ prime_arrivals, lane_order, priority options; a0 = y0 with prime_arrivals on; z_* = y0 with one change (prime_arrivals "
          "off except z_pa); a_pm0* = a0 with changes",
     "a0 f_final final1 z_pa z_lcost z_lcap z_rev z_none a_pm0 a_pm0_m1 a_pm0_m3 a_pm0_on8 a_pm0_on95"),
    ("G", "+ share_first and yield_ratio options (g0, g0b = a0)", "g0 g0b g_sf5 g_sf7 g_y5 g_y8 g_y10"),
    ("H", "ablations: a0 with ONE rule removed", "abl_nopulse abl_novalve abl_singlepass abl_notransit abl_nocapacity abl_nocrudefirst abl_nobank abl_nucmax"),
    ("I", "+ run_band and a dynamic end-of-horizon rule (lot_weeks, min_run); h0, k0 = a0; final2 = THE FINAL AGENT "
          "(run_band 0.9, share_first 0.5, the file handed in)",
     "h0 k0 h_rb95 h_rb9 h_rb8 h_rb6 k_d11m3 k_d12m2 k_d12m3 k_d12m4 k_d12m6 k_d13m3 m_rb92 m_sf3 m_rb9_sf4 m_rb9_sf5 final2"),
]
NOTE = {
    "abl_nopulse": "pulse=False: the valve always moves as the level asks (no hold/prime/run, no banking)",
    "abl_novalve": "pulse off and no closing-stock rule: the valve moves everything the terminal holds (up to the grid's storage)",
    "abl_singlepass": "top_weeks=0: no top-up pass",
    "abl_notransit": "inventory position = stock on hand only (cargo in transit and waiting at straits not counted)",
    "abl_nocapacity": "requests ignore the lane's edges, straits and the fleet slack: min(need, source stock)",
    "abl_nocrudefirst": "crude_first=False",
    "abl_nobank": "bank_crude=False",
    "abl_nucmax": "nuclear fuel: ask for all the lanes can carry, as pull does",
    "final2": "THE FINAL AGENT",
}
seen, lines = {}, []
for tag, desc, names in STAGES:
    lines.append(f"### {tag}. {desc}")
    lines.append("| variant | changes (params.json) | score | diff vs pull | 90% interval of the diff |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    merged = {}
    for n in names.split():
        if n not in rows:
            continue
        f = rows[n].split()
        pf = ROOT / "scratch" / "variants" / n / "params.json"
        p = json.loads(pf.read_text()) if pf.is_file() else {}
        ptxt = ", ".join(f"{k}={v}" for k, v in p.items()) or "(stage defaults)"
        if n in NOTE:
            ptxt = NOTE[n] + (f" [{ptxt}]" if p else "")
        key = (f[0], f[1], " ".join(f[2:5]), ptxt)
        merged.setdefault(key, []).append(n)
    for (score, diff, iv, ptxt), ns in merged.items():
        lines.append(f"| {' = '.join(ns)} | {ptxt} | {score} | {diff} | {iv} |")
    lines.append("")
text = "\n".join(lines)
(ROOT / "lab_out" / "records" / "table_variants_by_stage.txt").write_text(text + "\n")
print(text)
print(f"\n({sum(len(n.split()) for _, _, n in STAGES)} snapshot folders listed; {len(rows)} scored)")
