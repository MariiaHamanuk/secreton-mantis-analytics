"""Gate measurement and training tables for the imitation student.

For every teacher episode ``ep<n>.npz`` (format: ``../teacher/README.md``):

1. rebuild the week's observation dict from the stored ``obs/<field>`` arrays;
2. replay ``agents/anastasiia_hybrid_hub`` along that observation sequence (a fresh ``Agent`` per episode, ``act``
   every week) to get the ANCHOR action of the week;
3. featurize the week with ``rl_lab/features.py`` (the anchor flows fill the ``rules`` group);
4. save ``outputs/student/prep_<task>_<entropy>/ep<n>.npz`` with the tables, anchor and teacher actions, mask.

Then print the gate statistics over all weeks of all episodes (``gate``):

(a) share of the teacher's flow (raw units and value-weighted, v from config) on slots where the anchor sends 0;
(b) teacher/anchor ratios where both are > 0;
(c) share of weeks whose teacher ``release_mode`` differs from the anchor's, per release pair;
(d) mean |teacher - anchor| per slot in units of ``Layout.slot_cap0``.

    uv run python lab/anastasiia/mpc_lab/student/prep.py run --src=outputs/teacher/small_555
    uv run python lab/anastasiia/mpc_lab/student/prep.py gate --prep=outputs/student/prep_small_555

Runs in the plain project environment (numpy only); one process.
"""

import gzip
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RL_LAB = ROOT / "lab" / "anastasiia" / "rl_lab"
ANCHOR = ROOT / "agents" / "anastasiia_hybrid_hub"
LOG = HERE / "build.log"

sys.path.insert(0, str(RL_LAB))
from features import Featurizer, History, Layout  # noqa: E402


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_agent_module(folder, tag="anchor"):
    folder = Path(folder).resolve()
    spec = importlib.util.spec_from_file_location(f"{folder.name}_{tag}_agent", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(src):
    with gzip.open(Path(src) / "config.pkl.gz", "rb") as f:
        return pickle.load(f)


def episodes(src):
    return sorted(int(p.stem[2:]) for p in Path(src).glob("ep*.npz"))


def obs_seq(z):
    """The stored observations as a list of per-week dicts, exactly as ``Agent.act`` sees them."""
    keys = [k for k in z.files if k.startswith("obs/")]
    arrays = {k[4:]: z[k] for k in keys}
    T = arrays["week"].shape[0]
    return [{k: np.array(a[t]) for k, a in arrays.items()} for t in range(T)]


def process(src, out, ep, config, module, layout):
    z = np.load(Path(src) / f"ep{ep}.npz")
    obs = obs_seq(z)
    agent = module.Agent(config)
    feat = Featurizer(layout)
    hist = History(layout)
    node, edge, slot, glob = [], [], [], []
    a_flows, a_ovq, a_rm = [], [], []
    for o in obs:
        hist.update(layout, o)
        act = agent.act(o)
        fl = np.asarray(act["flows"], dtype=np.float64)
        a_flows.append(fl)
        a_ovq.append(np.asarray(act.get("override_qty", np.zeros(z["act/override_qty"].shape[1])), dtype=np.float64))
        a_rm.append(np.asarray(act.get("release_mode", np.zeros(z["act/release_mode"].shape[1])), dtype=np.int64))
        nd, ed, sl, gl = feat.week(o, fl, hist)
        node.append(nd)
        edge.append(ed)
        slot.append(sl)
        glob.append(gl)
    rec = {
        "node": np.stack(node),
        "edge": np.stack(edge),
        "slot": np.stack(slot),
        "global": np.stack(glob),
        "anchor_flows": np.stack(a_flows),
        "anchor_override_qty": np.stack(a_ovq),
        "anchor_release_mode": np.stack(a_rm),
        "teacher_flows": z["act/flows"],
        "teacher_override_qty": z["act/override_qty"],
        "teacher_release_mode": z["act/release_mode"],
        "action_mask": z["obs/action_mask"].astype(np.float32),
        "week": z["obs/week"].reshape(-1),
        "J_teacher": z["J_teacher"],
        "J_naive": z["J_naive"],
        "J_oracle": z["J_oracle"],
        "stratum": z["stratum"],
    }
    tmp = out / f"ep{ep}.tmp.npz"
    np.savez_compressed(tmp, **rec)
    tmp.rename(out / f"ep{ep}.npz")
    return rec


def run(src="outputs/teacher/small_555", out=None, force=False):
    src = ROOT / src if not Path(src).is_absolute() else Path(src)
    name = src.name
    out = Path(out) if out else ROOT / "outputs" / "student" / f"prep_{name}"
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(src)
    layout = Layout(config)
    module = load_agent_module(ANCHOR)
    eps = episodes(src)
    log(f"prep: {src.relative_to(ROOT)} -> {out.relative_to(ROOT)}, {len(eps)} episodes {eps}")
    for ep in eps:
        if (out / f"ep{ep}.npz").exists() and not force:
            continue
        t0 = time.process_time()
        rec = process(src, out, ep, config, module, layout)
        log(
            f"prep ep{ep}: T={rec['teacher_flows'].shape[0]} slot_feat={rec['slot'].shape[-1]} "
            f"node={rec['node'].shape[-1]} edge={rec['edge'].shape[-1]} glob={rec['global'].shape[-1]} "
            f"cpu={time.process_time() - t0:.1f}s"
        )
    gate(out, src)


def gate(prep="outputs/student/prep_small_555", src=None):
    prep = ROOT / prep if not Path(prep).is_absolute() else Path(prep)
    src = Path(src) if src else ROOT / "outputs" / "teacher" / prep.name.replace("prep_", "")
    config = load_config(src)
    layout = Layout(config)
    v = np.array(config["static"]["commodities"]["v"], dtype=np.float64)
    names = list(config["static"]["commodities"]["id"])
    vs = v[layout.slot_k]
    cap = layout.slot_cap0
    files = sorted(prep.glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    files = [p for p in files if not p.stem.endswith(".tmp")]
    A, Tt, M, RA, RT, OA, OT = [], [], [], [], [], [], []
    for p in files:
        z = np.load(p)
        A.append(z["anchor_flows"])
        Tt.append(z["teacher_flows"])
        M.append(z["action_mask"])
        RA.append(z["anchor_release_mode"])
        RT.append(z["teacher_release_mode"])
        OA.append(z["anchor_override_qty"])
        OT.append(z["teacher_override_qty"])
    A, Tt, M = np.concatenate(A), np.concatenate(Tt), np.concatenate(M)
    RA, RT, OA, OT = np.concatenate(RA), np.concatenate(RT), np.concatenate(OA), np.concatenate(OT)
    n_weeks = A.shape[0]
    tol = 1e-6
    a0, t0 = A <= tol, Tt <= tol
    res = {"episodes": len(files), "weeks": int(n_weeks)}
    lines = [f"episodes {len(files)}, weeks {n_weeks}, slots {A.shape[1]}"]

    # (a) teacher volume on anchor-zero slots
    tv, tvv = Tt.sum(), (Tt * vs).sum()
    res["a_share_vol"] = float(Tt[a0].sum() / tv)
    res["a_share_val"] = float((Tt * vs)[a0].sum() / tvv)
    res["a_slots_anchor_zero"] = float((a0 & (M > 0)).sum() / max((M > 0).sum(), 1))
    res["a_slots_teacher_pos_anchor_zero"] = float((a0 & ~t0).sum() / max((~t0).sum(), 1))
    avv = (A * vs).sum()
    res["a_anchor_on_teacher_zero_val"] = float((A * vs)[t0].sum() / avv)
    lines.append(
        f"(a) teacher flow on anchor-zero slots: {res['a_share_vol']:.3f} of volume, {res['a_share_val']:.3f} of value"
        f" | anchor-zero share of masked slot-weeks {res['a_slots_anchor_zero']:.3f}"
        f" | of teacher-positive slot-weeks {res['a_slots_teacher_pos_anchor_zero']:.3f}"
        f" | (mirror) anchor value on teacher-zero slots {res['a_anchor_on_teacher_zero_val']:.3f}"
    )
    per_k = {}
    for k, name in enumerate(names):
        sel = layout.slot_k == k
        tk = Tt[:, sel]
        ak = A[:, sel]
        if tk.sum() <= 0 and ak.sum() <= 0:
            continue
        per_k[name] = {
            "teacher_val_share_of_total": float((tk * v[k]).sum() / tvv),
            "teacher_on_anchor_zero": float(tk[ak <= tol].sum() / max(tk.sum(), tol)),
            "teacher_over_anchor_volume": float(tk.sum() / max(ak.sum(), tol)),
        }
        lines.append(
            f"    {name:13s} share of teacher value {per_k[name]['teacher_val_share_of_total']:.3f}; "
            f"on anchor-zero {per_k[name]['teacher_on_anchor_zero']:.3f}; "
            f"teacher/anchor total volume {per_k[name]['teacher_over_anchor_volume']:.3f}"
        )
    res["a_per_commodity"] = per_k

    # (b) ratios where both > 0
    both = ~a0 & ~t0
    r = Tt[both] / A[both]
    q = np.quantile(r, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    res["b_n"] = int(both.sum())
    res["b_quantiles"] = dict(zip(["5", "10", "25", "50", "75", "90", "95"], map(float, q)))
    res["b_out_0.5_2"] = float(((r < 0.5) | (r > 2.0)).mean())
    w = (np.maximum(Tt, A) * vs)[both]
    res["b_out_0.5_2_valw"] = float((w * ((r < 0.5) | (r > 2.0))).sum() / w.sum())
    res["b_within_5pct"] = float((np.abs(r - 1) <= 0.05).mean())
    lines.append(
        f"(b) both>0 slot-weeks {res['b_n']}; ratio quantiles 5/10/25/50/75/90/95% = "
        + " / ".join(f"{x:.3g}" for x in q)
        + f"; outside [0.5,2]: {res['b_out_0.5_2']:.3f} (value-weighted {res['b_out_0.5_2_valw']:.3f});"
        f" within ±5%: {res['b_within_5pct']:.3f}"
    )

    # (c) release mode
    pairs = config["layout"]["release_pairs"]
    diff = RA != RT
    res["c_share_diff"] = float(diff.mean())
    res["c_per_pair"] = []
    lines.append(
        f"(c) release_mode differs in {res['c_share_diff']:.3f} of pair-weeks; "
        f"anchor modes {np.bincount(RA.ravel(), minlength=3).tolist()} teacher modes "
        f"{np.bincount(RT.ravel(), minlength=3).tolist()} (default/override/hold)"
    )
    chk_names = config["static"]["nodes"]["id"]
    for i, (c, k) in enumerate(pairs):
        row = {
            "pair": f"{chk_names[int(c)]}/{names[int(k)]}",
            "diff": float(diff[:, i].mean()),
            "anchor_override": float((RA[:, i] == 1).mean()),
            "teacher_override": float((RT[:, i] == 1).mean()),
            "teacher_hold": float((RT[:, i] == 2).mean()),
        }
        res["c_per_pair"].append(row)
        lines.append(
            f"    {row['pair']:28s} differs {row['diff']:.3f}  override: anchor {row['anchor_override']:.3f} "
            f"teacher {row['teacher_override']:.3f}  teacher hold {row['teacher_hold']:.3f}"
        )
    # override quantity in weeks the teacher overrides
    ov = config["static"]["override_slots"]
    ov_pair = [
        pairs.index([int(c), int(k)]) if [int(c), int(k)] in pairs else -1 for c, k in zip(ov["chokepoint"], ov["k"])
    ]
    ov_pair = np.array(ov_pair)
    t_ov = np.where(RT[:, ov_pair] == 1, OT, 0.0)
    a_ov = np.where(RA[:, ov_pair] == 1, OA, 0.0)
    vo = v[np.array([int(k) for k in ov["k"]])]
    res["c_override_value_teacher"] = float((t_ov * vo).sum() / len(files))
    res["c_override_value_anchor"] = float((a_ov * vo).sum() / len(files))
    res["c_flows_value_teacher"] = float(tvv / len(files))
    lines.append(
        f"    override_qty value per episode: teacher {res['c_override_value_teacher']:.3g} USD-v, anchor "
        f"{res['c_override_value_anchor']:.3g}; teacher flows value per episode {res['c_flows_value_teacher']:.3g}"
    )

    # (d) |teacher - anchor| per slot in slot_cap0 units
    d = np.abs(Tt - A) / cap[None, :]
    live = M > 0
    res["d_mean_all_masked"] = float(d[live].mean())
    nz = live & ~(a0 & t0)
    res["d_mean_nonzero"] = float(d[nz].mean())
    res["d_quantiles_nonzero"] = dict(
        zip(["50", "75", "90", "95", "99"], map(float, np.quantile(d[nz], [0.5, 0.75, 0.9, 0.95, 0.99])))
    )
    res["d_max"] = float(d.max())
    sd = (Tt - A) / cap[None, :]
    res["d_signed_mean_nonzero"] = float(sd[nz].mean())
    lines.append(
        f"(d) |teacher-anchor|/cap0: mean over masked slot-weeks {res['d_mean_all_masked']:.4f}; over not-both-zero "
        f"{res['d_mean_nonzero']:.4f} (signed {res['d_signed_mean_nonzero']:+.4f}); quantiles 50/75/90/95/99% "
        + " / ".join(f"{x:.3g}" for x in res["d_quantiles_nonzero"].values())
        + f"; max {res['d_max']:.3g}"
    )
    tc, ac = Tt / cap[None, :], A / cap[None, :]
    res["d_teacher_cap_q"] = list(map(float, np.quantile(tc[~t0], [0.5, 0.9, 0.99])))
    res["d_anchor_cap_q"] = list(map(float, np.quantile(ac[~a0], [0.5, 0.9, 0.99])))
    lines.append(
        f"    teacher/cap0 where >0, q50/90/99: {res['d_teacher_cap_q']}; anchor/cap0 where >0: {res['d_anchor_cap_q']}"
    )
    for k, name in enumerate(names):
        sel = (layout.slot_k == k)[None, :] & nz
        if sel.sum():
            lines.append(f"    {name:13s} mean |d|/cap0 {d[sel].mean():.4f}  (n={int(sel.sum())})")
    (prep / "gate.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    log("gate:\n  " + "\n  ".join(lines))
    return res


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run, "gate": gate})
