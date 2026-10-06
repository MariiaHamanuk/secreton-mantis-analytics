"""Scratch: where chips (and wafers) are disposed of, per node and commodity, from flow balance on an episode record.

    python lab_scratch/loss.py AGENT N [first]
"""
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "lab_scratch")


def disposal_by_slot(r):
    T = r.T
    obs = r.obs
    S = len(r.stock_slots)
    stock = np.asarray(obs["stock.qty"])  # (T+1, S)
    slot_of = r.slot_of
    st = r.static
    E = st["edges"]
    sl = st["action_slots"]
    # executed outflow per stock slot per week (dispatch from the tail node)
    exe = np.asarray(obs["last_week.clip.executed"])  # row t = week t's
    out = np.zeros((T + 1, S))
    for s in range(len(sl["edge"])):
        tail = E["tail"][sl["edge"][s]]
        k = sl["k"][s]
        if (tail, k) in slot_of:
            out[:, slot_of[(tail, k)]] += exe[:, s]
    # arrivals per week at head nodes: pipeline entries of the obs at week t with arrival_week == t
    arr = np.zeros((T + 1, S))
    for t in range(1, T + 1):
        live = np.asarray(obs["pipeline.qty.observed"][t - 1]) == 1
        e = np.asarray(obs["pipeline.edge"][t - 1])[live]
        k = np.asarray(obs["pipeline.k"][t - 1])[live]
        q = np.asarray(obs["pipeline.qty"][t - 1])[live]
        aw = np.asarray(obs["pipeline.arrival_week"][t - 1])[live]
        for ei, ki, qi, wi in zip(e, k, q, aw):
            if int(wi) == t:
                head = E["head"][int(ei)]
                if (head, int(ki)) in slot_of:
                    arr[t, slot_of[(head, int(ki))]] += qi
    # production at fabs and OSATs: WIP entries with out_week == t in obs of week t (index t-1)
    prod = np.zeros((T + 1, S))
    for t in range(1, T + 1):
        live = np.asarray(obs["wip.qty.observed"][t - 1]) == 1
        for nd, kk, q, ow in zip(np.asarray(obs["wip.node"][t - 1])[live], np.asarray(obs["wip.k"][t - 1])[live], np.asarray(obs["wip.qty"][t - 1])[live], np.asarray(obs["wip.out_week"][t - 1])[live]):
            if int(ow) == t and (int(nd), int(kk)) in slot_of:
                prod[t, slot_of[(int(nd), int(kk))]] += q
    # consumption: fab starts (wafer), osat packaging starts (raw), market serving
    cons = np.zeros((T + 1, S))
    for fab in r.fabs:
        name = r.node[fab]
        a = r.fab_attr(name)
        p = r.wip_starts(name)
        sidx = slot_of[(fab, r.com_ix[a["input"]])]
        cons[1:, sidx] += p
    for o in r.osats:
        # packaging starts in week t: OSAT WIP entries with out_week == t + tau seen in obs of week t+1 (index t)
        tau = r.nodes_attr[o]["osat"]["tau"]
        raw_of = r.nodes_attr[o]["osat"]["packages"]
        for t in range(1, T + 1):
            live = np.asarray(obs["wip.qty.observed"][t]) == 1
            for nd, kk, q, ow in zip(np.asarray(obs["wip.node"][t])[live], np.asarray(obs["wip.k"][t])[live], np.asarray(obs["wip.qty"][t])[live], np.asarray(obs["wip.out_week"][t])[live]):
                if int(nd) == o and int(ow) == t + tau:
                    pk = r.com[int(kk)]
                    raw = [rw for rw, pkn in raw_of.items() if pkn == pk][0]
                    cons[t, slot_of[(o, r.com_ix[raw])]] += q
    served = np.asarray(obs["last_week.sinks.served"])  # row t = week t
    for di, (nd, k) in enumerate(r.demands):
        cons[:, slot_of[(nd, k)]] += served[:, di]
    disp = np.zeros((T + 1, S))
    for t in range(1, T + 1):
        disp[t] = stock[t - 1] - out[t] + arr[t] + prod[t] - cons[t] - stock[t]
    return disp, stock, out, arr, prod, cons


def run(agent, n):
    from rec import play
    from dec import Rec

    r = Rec(play(agent, n))
    disp, *_ = disposal_by_slot(r)
    return disp[1:].sum(axis=0), r


def main(agent, n=6, first=0, n_jobs=3):
    from dec import Rec

    res = Parallel(n_jobs=n_jobs)(delayed(lambda a, e: run(a, e)[0])(agent, e) for e in range(first, first + n))
    # names
    from rec import play
    import pickle

    tot = np.mean(res, axis=0)
    # stock slot labels from any record: rebuild cheaply using the static of episode 0
    r0 = run(agent, first)[1]
    rows = []
    for i, (nd, k) in enumerate(r0.stock_slots):
        if tot[i] > 1000:
            rows.append((tot[i], r0.node[nd], r0.com[k]))
    for q, nd, k in sorted(rows, reverse=True):
        print(f"{nd:20s} {k:13s} disposed per episode {q / 1e3:10,.0f}k")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 6, int(a[3]) if len(a) > 3 else 0)
