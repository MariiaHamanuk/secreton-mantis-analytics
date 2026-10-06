import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
names = ["freight", "war", "tariff", "holding", "queue", "shortage", "disposal", "shed"]
print(f"{'agent':8s} " + " ".join(f"{n:>9s}" for n in names) + "     total")
for tag in tags:
    tot = np.zeros(8)
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        tot += r.costs.sum(axis=0)
    tot /= len(eps) * 1e9
    print(f"{tag:8s} " + " ".join(f"{v:9.1f}" for v in tot) + f"  {tot.sum():8.1f}")
