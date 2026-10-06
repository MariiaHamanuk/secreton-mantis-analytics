"""Scratch: play episodes of an agent and pickle each record: python save_recs.py AGENT TAG N [first]"""
import pickle
import sys

from joblib import Parallel, delayed

sys.path.insert(0, "lab_scratch")


def one(agent, tag, n):
    from rec import play

    rec = play(agent, n)
    with open(f"lab_scratch/recs/{tag}_{n}.pkl", "wb") as f:
        pickle.dump(rec, f)
    return n


if __name__ == "__main__":
    import os

    os.makedirs("lab_scratch/recs", exist_ok=True)
    a = sys.argv
    agent, tag, n = a[1], a[2], int(a[3])
    first = int(a[4]) if len(a) > 4 else 0
    Parallel(n_jobs=3)(delayed(one)(agent, tag, e) for e in range(first, first + n))
    print("done")
