"""A paired difference over episodes played on two machines, each episode against the base of its own machine.

    uv run python lab/anastasiia/frontier_lab/mech/pooled.py "h3_s:0-63,h3c_s:64-127" "h3p3_s:0-63,h3p3c_s:64-127" two8b

The first argument is the base, every other one a variant; each is one tag, or tags with the episodes taken from
each ("tag:a-b,tag:c-d"). The score and the paired interval are ``hazard_lab/play.py show``'s (episodes drawn again
inside the harm levels, 2,000 times, the 5th to the 95th percentile).
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scen"), str(HERE.parents[1] / "hazard_lab")]


def _costs(spec: str, task: str, entropy: int) -> dict:
    import play as P

    out = {}
    for part in str(spec).split(","):
        tag, _, eps = part.partition(":")
        kept = P.kept(tag, task, entropy)
        for n in (P._numbers(eps) if eps else sorted(kept)):
            if n in kept:
                out[n] = kept[n]["J"]
    return out


def main(base: str, *variants: str, task: str = "small", entropy: int = 444, draws: int = 2000) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("hazard_play", HERE.parents[1] / "hazard_lab" / "play.py")
    hz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hz)
    a = _costs(base, task, entropy)
    refs_all = hz.references(task, entropy, max(a) + 1)
    for variant in variants:
        b = _costs(variant, task, entropy)
        ns = [n for n in sorted(a) if n in b and refs_all[n]["J_oracle_cents"] is not None]
        refs = [refs_all[n] for n in ns]
        level = np.array([r["stratum"] for r in refs])
        ca, cb = np.array([a[n] for n in ns], dtype=float), np.array([b[n] for n in ns], dtype=float)
        groups = [np.flatnonzero(level == s) for s in sorted(set(level))]
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(draws):
            p = np.concatenate([rng.choice(g, len(g)) for g in groups])
            rp = [refs[i] for i in p]
            diffs.append(hz.rss(rp, cb[p])[0] - hz.rss(rp, ca[p])[0])
        lo, hi = np.percentile(diffs, [5, 95])
        d = ca - cb
        print(f"{variant} to {base}: {len(ns)} episodes ({ns[0]}..{ns[-1]}), {hz.rss(refs, cb)[0]:.4f} against {hz.rss(refs, ca)[0]:.4f}: "
              f"{hz.rss(refs, cb)[0] - hz.rss(refs, ca)[0]:+.4f} ({lo:+.4f} to {hi:+.4f}), cheaper in {int((d > 0).sum())} of {len(ns)}, "
              f"{d.mean() / 1e11:+.1f} bn an episode")


if __name__ == "__main__":
    fire.Fire(main)
