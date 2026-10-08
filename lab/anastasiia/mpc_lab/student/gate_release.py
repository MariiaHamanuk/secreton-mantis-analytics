"""Gate, part 3: is the teacher's strait-release gain state-dependent or foresight rent?

The anchor (``agents/anastasiia_hybrid_hub``) plays its own flows closed loop; only the release side
(``release_mode``, ``override_qty``) is replaced, open loop, by:

- ``own``         the teacher's schedule of the same episode (``a_flows+t_rel`` of ``gate_play.py``);
- ``shift+1``     the teacher's schedule one week late (week t plays the teacher's t-1; week 0 the anchor's rules);
- ``shift-1``     one week early (week t plays t+1; the last week the anchor's rules);
- ``foreign``     the schedule of episode ``eps[(i+1) % len(eps)]``, week by week;
- ``alpha=a``     constant metering: every pair with cargo queued at the week's start is switched to override, and
                  each queued lot group (strait, k, lane, next edge) releases ``a`` x its queue onto its own lane's
                  override slot (``lot_keys`` -> ``override_slots`` match 35 of 35 fuel keys on Small). A pair with
                  nothing queued keeps the default release. ``alpha_k=lng:a,crude:b`` sets it per commodity.

``stats`` prints how the teacher's alpha = released / queued varies per pair and its rank correlation with what
the observation shows (queue size, week, strait open, tanker kappa at the strait, capacity of the next edges).

    uv run python lab/anastasiia/mpc_lab/student/gate_release.py run "--variants=own|shift+1|shift-1|foreign"
    uv run python lab/anastasiia/mpc_lab/student/gate_release.py run "--variants=alpha=0.25|alpha=0.4"
    uv run python lab/anastasiia/mpc_lab/student/gate_release.py stats
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from gate_play import play  # noqa: E402
from prep import ANCHOR, load_agent_module, load_config, log  # noqa: E402


class Meter:
    """Constant metering of the strait queues: ``alpha[k]`` x what waits at the week's start, on its own lane."""

    def __init__(self, config, alpha):
        lay, ov = config["layout"], config["static"]["override_slots"]
        self.pairs = [(int(c), int(k)) for c, k in lay["release_pairs"]]
        self.lot_keys = [tuple(int(x) for x in key) for key in lay["lot_keys"]]
        slot_of = {
            (int(c), int(k), int(e), None if ln is None else int(ln)): s
            for s, (c, k, e, ln) in enumerate(zip(ov["chokepoint"], ov["k"], ov["out_edge"], ov["lane"]))
        }
        self.n_ov = len(ov["chokepoint"])
        self.row_slot = [slot_of.get((c, k, nxt, lane), -1) for c, k, lane, nxt in self.lot_keys]
        self.alpha = alpha  # dict k -> alpha

    def fill(self, obs):
        qty = np.zeros(self.n_ov)
        mode = np.zeros(len(self.pairs), dtype=np.int64)
        if "override_mask.observed" in obs and int(np.asarray(obs["override_mask.observed"])[0]) != 1:
            return {"override_qty": qty, "release_mode": mode}
        omask = np.asarray(obs["override_mask"]) == 1
        q = np.where(np.asarray(obs["queue_lots.qty.observed"]) == 1, obs["queue_lots.qty"], 0.0).sum(axis=1)
        for i, (c, k) in enumerate(self.pairs):
            a = self.alpha.get(k)
            if a is None:
                continue
            rows = [r for r, key in enumerate(self.lot_keys) if key[0] == c and key[1] == k and q[r] > 0]
            if not rows or any(self.row_slot[r] < 0 or not omask[self.row_slot[r]] for r in rows):
                continue  # nothing queued, or a lot group we cannot address: the default release
            mode[i] = 1
            for r in rows:
                qty[self.row_slot[r]] += a * float(q[r])
        return {"override_qty": qty, "release_mode": mode}


def parse_alpha(spec, names):
    if spec.startswith("alpha="):
        a = float(spec.split("=", 1)[1])
        return {k: a for k, n in enumerate(names) if n in ("lng", "crude")}
    # alpha_k=lng:0.4,crude:0.6
    out = {}
    for part in spec.split("=", 1)[1].split(","):
        n, a = part.split(":")
        out[names.index(n)] = float(a)
    return out


def run(task="small", entropy=555, episodes="0,1,2,3,4,5,6,7,8,9", variants="own", tag=None):
    src = ROOT / "outputs" / "teacher" / f"{task}_{entropy}"
    eps = [int(x) for x in str(episodes).split(",")] if isinstance(episodes, str) else list(episodes)
    # variants are separated by "|" (a per-commodity alpha holds commas); fire may hand a tuple
    vs = list(variants) if isinstance(variants, (list, tuple)) else [v for v in str(variants).split("|") if v]
    config = load_config(src)
    names = list(config["static"]["commodities"]["id"])
    module = load_agent_module(ANCHOR)
    sched = {}
    for n in eps:
        z = np.load(src / f"ep{n}.npz")
        sched[n] = (z["act/override_qty"], z["act/release_mode"])
    rows = []
    for i, n in enumerate(eps):
        t0 = time.time()
        hub = module.Agent(config)
        out = {"ep": n, "anchor": play(task, entropy, n, lambda t, o, hub=hub: hub.act(o))[0]}
        for v in vs:
            hub = module.Agent(config)
            if v.startswith("alpha"):
                meter = Meter(config, parse_alpha(v, names))

                def pol(t, o, hub=hub, meter=meter):
                    return {"flows": hub.act(o)["flows"], **meter.fill(o)}
            else:
                if v == "own":
                    OQ, RM, s = *sched[n], 0
                elif v.startswith("shift"):
                    OQ, RM, s = *sched[n], int(v[5:])
                elif v == "foreign":
                    OQ, RM, s = *sched[eps[(i + 1) % len(eps)]], 0
                else:
                    raise ValueError(v)

                def pol(t, o, hub=hub, OQ=OQ, RM=RM, s=s):
                    a = hub.act(o)
                    j = t - s
                    if 0 <= j < len(RM):
                        return {"flows": a["flows"], "override_qty": OQ[j], "release_mode": RM[j]}
                    return a  # outside the schedule: the anchor's own strait rules

            out[v] = play(task, entropy, n, pol)[0]
        rows.append(out)
        log(
            f"gate_release ep{n}: bn vs anchor "
            + ", ".join(f"{v} {(out['anchor'] - out[v]) / 1e11:+.1f}" for v in vs)
            + f" ({time.time() - t0:.0f}s)"
        )
    summ = {}
    for v in vs:
        g = np.array([(r["anchor"] - r[v]) / 1e11 for r in rows])
        summ[v] = {
            "gain_bn": float(g.mean()),
            "se": float(g.std(ddof=1) / np.sqrt(len(g))),
            "better": int((g > 0).sum()),
            "n": len(g),
        }
    name = tag or "_".join(v.replace("=", "").replace(":", "").replace(",", "") for v in vs)[:80]
    (ROOT / "outputs" / "student" / f"gate_release_{task}_{entropy}_{name}.json").write_text(
        json.dumps({"rows": rows, "summary": summ}, indent=2)
    )
    log("gate_release summary (bn/ep vs anchor, + better): " + json.dumps(summ))
    return summ


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def stats(task="small", entropy=555):
    """The teacher's alpha = released / queued per pair-week, and what it moves with."""
    src = ROOT / "outputs" / "teacher" / f"{task}_{entropy}"
    config = load_config(src)
    lay, ov, st = config["layout"], config["static"]["override_slots"], config["static"]
    names = list(st["commodities"]["id"])
    nodes = st["nodes"]["id"]
    pairs = [(int(c), int(k)) for c, k in lay["release_pairs"]]
    lot_keys = [tuple(int(x) for x in key) for key in lay["lot_keys"]]
    chk_row = {int(c): i for i, c in enumerate(lay["chokepoints"])}
    u0 = np.array([np.inf if x is None else float(x) for x in st["edges"]["u0"]])
    ov_pair = [(int(c), int(k)) for c, k in zip(ov["chokepoint"], ov["k"])]
    data = defaultdict(lambda: defaultdict(list))
    files = sorted(src.glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    for p in files:
        z = np.load(p)
        q = np.where(z["obs/queue_lots.qty.observed"] == 1, z["obs/queue_lots.qty"], 0.0).sum(axis=2)
        OQ, RM = z["act/override_qty"], z["act/release_mode"]
        u = z["obs/graph_now.u"]
        for t in range(q.shape[0]):
            for i, (c, k) in enumerate(pairs):
                rows = [r for r, key in enumerate(lot_keys) if key[0] == c and key[1] == k]
                queued = float(q[t, rows].sum())
                if RM[t, i] != 1 or queued <= 1e-6:
                    continue
                rel = float(sum(OQ[t, s] for s, pk in enumerate(ov_pair) if pk == (c, k)))
                nxt_u = [
                    min(float(u[t, lot_keys[r][3]]) / u0[lot_keys[r][3]], 2.0)
                    if np.isfinite(u0[lot_keys[r][3]])
                    else 1.0
                    for r in rows
                    if q[t, r] > 0
                ]
                d = data[(c, k)]
                d["alpha"].append(rel / queued)
                d["queue"].append(queued)
                d["week"].append(t + 1)
                d["open"].append(float(z["obs/graph_now.open"][t, chk_row[c]]))
                d["kappa"].append(float(z["obs/graph_now.kappa.tb"][t, chk_row[c]]))
                d["next_u"].append(float(np.mean(nxt_u)) if nxt_u else 1.0)
                d["ep"].append(int(p.stem[2:]))
    lines = [f"teacher alpha = released / queued at week start, {len(files)} episodes; Spearman rho with observables"]
    lines.append(
        f"{'pair':22s} {'n':>4s} {'med':>6s} {'q25':>6s} {'q75':>6s}  {'queue':>6s} {'week':>6s} {'open':>6s} "
        f"{'kappa':>6s} {'next_u':>6s}  {'queue/ep':>8s}"
    )
    pooled = defaultdict(list)
    out = {}
    for (c, k), d in sorted(data.items()):
        a = np.array(d["alpha"])
        qn = np.array(d["queue"]) / np.median(d["queue"])
        # within-episode queue rank: does the teacher release a bigger share when the queue is big for this episode?
        qe = np.array(d["queue"]) / np.array(
            [np.median([x for x, e in zip(d["queue"], d["ep"]) if e == ep]) for ep in d["ep"]]
        )
        rho = {
            "queue": spearman(a, qn),
            "week": spearman(a, d["week"]),
            "open": spearman(a, d["open"]),
            "kappa": spearman(a, d["kappa"]),
            "next_u": spearman(a, d["next_u"]),
            "queue_in_ep": spearman(a, qe),
        }
        key = f"{nodes[c]}/{names[k]}"
        out[key] = {
            "n": len(a),
            "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
            "rho": rho,
        }
        lines.append(
            f"{key:22s} {len(a):4d} {np.median(a):6.3f} {np.quantile(a, 0.25):6.3f} {np.quantile(a, 0.75):6.3f}  "
            + " ".join(f"{rho[x]:+6.2f}" for x in ("queue", "week", "open", "kappa", "next_u"))
            + f"  {rho['queue_in_ep']:+8.2f}"
        )
        pooled["alpha"] += list(a)
        pooled["queue"] += list(qn)
        for x in ("week", "open", "kappa", "next_u"):
            pooled[x] += d[x]
    a = np.array(pooled["alpha"])
    lines.append(
        f"{'pooled':22s} {len(a):4d} {np.median(a):6.3f} {np.quantile(a, 0.25):6.3f} {np.quantile(a, 0.75):6.3f}  "
        + " ".join(f"{spearman(a, pooled[x]):+6.2f}" for x in ("queue", "week", "open", "kappa", "next_u"))
    )
    # how much of the variance is between episodes vs within an episode (per pair, pooled)
    within, total = 0.0, 0.0
    for d in data.values():
        a = np.log(np.clip(np.array(d["alpha"]), 1e-3, None))
        ep = np.array(d["ep"])
        total += ((a - a.mean()) ** 2).sum()
        within += sum(((a[ep == e] - a[ep == e].mean()) ** 2).sum() for e in np.unique(ep))
    lines.append(f"share of log-alpha variance within episodes (per pair): {within / total:.2f}")
    out["_within_share"] = within / total
    (ROOT / "outputs" / "student" / f"gate_release_stats_{task}_{entropy}.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))
    log("gate_release stats:\n  " + "\n  ".join(lines))
    return out


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run, "stats": stats})
