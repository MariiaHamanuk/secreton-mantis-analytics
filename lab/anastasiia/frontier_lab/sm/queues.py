"""Container queues at straits in kept plays: unit-weeks by strait and commodity, what stands there at the end."""
import pickle
import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
for tag, task in [a.split(":") for a in sys.argv[1:]]:
    inst, _ = task_generator(task)
    kept = pickle.load(open(f"{ROOT}/outputs/hazard_lab/play/{tag}_{task}_444.pkl", "rb"))
    C = [c.id for c in inst.commodities]
    ct = [k for k, c in enumerate(inst.commodities) if c.pool == "ct"]
    slots = {(inst.nodes[c].id, C[k]): inst.slot_index[(c, k)] for c in inst.chokepoints for k in ct if (c, k) in inst.slot_index}
    T = inst.T
    ns = sorted(kept)
    print(f"\n=== {tag} {task}: {len(ns)} episodes")
    tot, end, mx = {}, {}, {}
    per_ep = {}
    for n in ns:
        st = kept[n]["stock"]
        assert st.shape[0] == T, st.shape
        for key, s in slots.items():
            q = st[:, s]
            tot.setdefault(key, []).append(q.sum())
            end.setdefault(key, []).append(q[T - 5])  # a raw chip here can still be packaged and sold
            mx.setdefault(key, []).append(q.max())
        per_ep[n] = {key: st[:, s].sum() for key, s in slots.items()}
    print("mean unit-weeks an episode, thousand (max episode) | standing at week T-4, thousand (max episode):")
    for key in sorted(slots):
        a, b = np.array(tot[key]), np.array(end[key])
        if a.mean() > 2e3:
            print(f"   {key[0]:12s} {key[1]:13s} {a.mean()/1e3:9.0f} ({a.max()/1e3:9.0f}, ep {ns[int(a.argmax())]}) | {b.mean()/1e3:8.1f} ({b.max()/1e3:8.0f}, ep {ns[int(b.argmax())]})  episodes over 1 mn unit-weeks: {int((a > 1e6).sum())}")
    le = np.zeros(len(ns)); mat = np.zeros(len(ns)); waf = np.zeros(len(ns))
    le_end = np.zeros(len(ns))
    for key in slots:
        if key[1] == "chip_le_raw":
            le += np.array(tot[key]); le_end += np.array(end[key])
        elif key[1] in ("chip_mat_raw", "chip_mat"):
            mat += np.array(tot[key])
        elif key[1] == "wafer":
            waf += np.array(tot[key])
    print(f"   chip_le_raw in strait queues: mean {le.mean()/1e3:.0f}k unit-weeks, median {np.median(le)/1e3:.0f}k, episodes over 0.5 mn: {int((le > 5e5).sum())}, top five: {[(ns[i], round(le[i]/1e3)) for i in np.argsort(-le)[:5]]}")
    print(f"   chip_le_raw standing at week T-4: mean {le_end.mean()/1e3:.1f}k = {le_end.mean()*50400/1e9:.2f} bn at the penalty; top five: {[(ns[i], round(le_end[i]/1e3)) for i in np.argsort(-le_end)[:5]]}")
    print(f"   wafers: mean {waf.mean()/1e3:.0f}k unit-weeks, top five: {[(ns[i], round(waf[i]/1e3)) for i in np.argsort(-waf)[:5]]}")
    print(f"   mature chips (raw and packaged): mean {mat.mean()/1e3:.0f}k unit-weeks")
    if 7 in per_ep:
        print("   episode 7:", {k: round(v / 1e3) for k, v in per_ep[7].items() if v > 5e3})
