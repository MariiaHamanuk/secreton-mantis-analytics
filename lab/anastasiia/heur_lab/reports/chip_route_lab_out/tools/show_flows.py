import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

rec = pickle.load(open(sys.argv[1], "rb"))
slots_arg = [int(x) for x in sys.argv[2].split(",")]
r = Rec(rec)
T = r.T
st = r.static
sl = st["action_slots"]
E = st["edges"]
u = np.asarray(r.obs["graph_now.u"])
req = np.asarray(r.obs["last_week.clip.requested"])  # row t = requested in week t-1
exe = np.asarray(r.obs["last_week.clip.executed"])
print("slot: edge | commodity")
for s in slots_arg:
    print(f"  slot {s}: {E['id'][sl['edge'][s]]} {r.com[sl['k'][s]]}")
print("week: for each slot: u(edge, as observed) / asked / executed (k)")
for t in range(1, T + 1):
    row = []
    for s in slots_arg:
        e = sl["edge"][s]
        row.append(f"{u[t-1, e]/1e3:6.0f}/{req[t, s]/1e3:6.0f}/{exe[t, s]/1e3:6.0f}")
    print(f"{t:2d} " + "  ".join(row))
