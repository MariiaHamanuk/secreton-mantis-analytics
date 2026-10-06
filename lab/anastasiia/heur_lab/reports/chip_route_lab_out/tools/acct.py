"""Scratch: mass balance of the chip chain for an agent over several episodes (root 111).

    python lab_scratch/acct.py AGENT_FOLDER N_EPISODES [first]

Prints per chip family (le, mat): chips started, served, implied disposal+scrap (start - served - change of the system's
content), and where the content sits at the end. Everything is per episode on average.
"""
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "lab_scratch")


def run(agent: str, n: int):
    from rec import play
    from dec import Rec

    rec = play(agent, n)
    r = Rec(rec)
    T = r.T
    obs = r.obs
    com = r.com
    ci = r.com_ix
    fam = {"le": [ci["chip_le_raw"], ci["chip_le"]], "mat": [ci["chip_mat_raw"], ci["chip_mat"]]}
    famk = {k: f for f, ks in fam.items() for k in ks}
    # content per family at each observation 1..T+1
    stock = np.asarray(obs["stock.qty"])  # (T+1, 59)
    sk = np.array([k for (_n, k) in r.stock_slots])
    sn = np.array([nd for (nd, _k) in r.stock_slots])
    content = {f: np.zeros(T + 1) for f in fam}
    where = {f: {} for f in fam}
    for f, ks in fam.items():
        for kk in ks:
            m = sk == kk
            content[f] += stock[:, m].sum(axis=1)
    # wip
    for t in range(T + 1):
        live = np.asarray(obs["wip.qty.observed"][t]) == 1
        for nd, kk, q in zip(np.asarray(obs["wip.node"][t])[live], np.asarray(obs["wip.k"][t])[live], np.asarray(obs["wip.qty"][t])[live]):
            if int(kk) in famk:
                content[famk[int(kk)]][t] += q
        lv = np.asarray(obs["pipeline.qty.observed"][t]) == 1
        for kk, q in zip(np.asarray(obs["pipeline.k"][t])[lv], np.asarray(obs["pipeline.qty"][t])[lv]):
            if int(kk) in famk:
                content[famk[int(kk)]][t] += q
    # queue lots: layout lot_keys -> commodity
    lk = r.layout["lot_keys"]
    qk = []
    for key in lk:
        c, k, *_ = key if not isinstance(key, str) else key.split("/")
        qk.append(k)
    qty = np.asarray(obs["queue_lots.qty"])  # (T+1, rows, weeks)
    qobs = np.asarray(obs["queue_lots.qty.observed"])
    qsum = (qty * (qobs == 1)).sum(axis=2)  # (T+1, rows)
    for row, k in enumerate(qk):
        kk = ci[k] if isinstance(k, str) else int(k)
        if kk in famk:
            content[famk[kk]] += qsum[:, row]
    # starts per week per family
    starts = {f: np.zeros(T) for f in fam}
    for fab in r.fabs:
        name = r.node[fab]
        prod = r.fab_attr(name)["product"]
        f = "le" if prod == "chip_le_raw" else "mat"
        starts[f] += r.wip_starts(name)
    served = np.asarray(obs["last_week.sinks.served"])[1:]  # (T, 8)
    dem_k = [k for (_n, k) in r.demands]
    sv = {"le": 0.0, "mat": 0.0}
    for di, k in enumerate(dem_k):
        sv["le" if com[k] == "chip_le" else "mat"] += served[:, di].sum()
    out = {}
    for f in fam:
        implied = starts[f].sum() - sv[f] - (content[f][T] - content[f][0])
        out[f] = dict(content0=content[f][0], contentT=content[f][T], starts=starts[f].sum(), served=sv[f], implied_loss=implied)
    out["cost"] = r.costs.sum(axis=0)
    return out


def main(agent: str, n: int = 6, first: int = 0, n_jobs: int = 3):
    res = Parallel(n_jobs=n_jobs)(delayed(run)(agent, e) for e in range(first, first + n))
    for f in ("le", "mat"):
        print(f"--- chip family {f} (units per episode, mean over {n} episodes, millions)")
        for key in ("content0", "starts", "served", "implied_loss", "contentT"):
            print(f"   {key:13s} {np.mean([x[f][key] for x in res]) / 1e6:9.2f}")
    c = np.mean([x["cost"] for x in res], axis=0) / 1e9
    print("costs bn:", dict(zip(["freight", "war", "tariff", "holding", "queue", "shortage", "disposal", "shed"], c.round(1))))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 6, int(a[3]) if len(a) > 3 else 0)
