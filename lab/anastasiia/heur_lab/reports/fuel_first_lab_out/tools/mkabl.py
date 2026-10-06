"""Ablation snapshots of the final agent: each removes one rule by a small text edit (scratch).

    python scratch/mkabl.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "agents" / "fuel_first" / "agent.py").read_text()


def make(name, edits=(), params=None):
    text = SRC
    for old, new in edits:
        assert text.count(old) == 1, (name, old[:60], text.count(old))
        text = text.replace(old, new)
    folder = ROOT / "scratch" / "variants" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(text)
    (folder / "params.json").write_text(json.dumps(params or {}))
    print(folder)


# no time concentration
make("abl_nopulse", params={"pulse": False})
# no crude banking / no crude first / single pass (no top-up pass)
make("abl_nobank", params={"bank_crude": False})
make("abl_nocrudefirst", params={"crude_first": False})
make("abl_singlepass", params={"top_weeks": 0.0})
# pull's rule for nuclear fuel: ask for everything the lane can carry
make("abl_nucmax", edits=[(
    "                    need = max(routine - position[(g, k)], short if short * P[\"nuc_safety\"] >= carry else 0.0)",
    "                    need = 1e12",
)])
# no closing-stock rule at the valve: move everything the terminal holds (up to the grid's storage)
make("abl_novalve", edits=[(
    "                move = max(0.0, end + burn - s0 - landing[(g, k)])\n",
    "                move = max(0.0, 0.98 * store_g - s0 - landing[(g, k)] + burn)\n",
)], params={"pulse": False})
# ignore what the lanes can carry this week (edges, straits, fleet slack): ask for min(need, source stock)
make("abl_nocapacity", edits=[(
    "            r = min(edge_left[x] for x in d[\"route\"])\n            for c in d[\"chk\"]:\n                r = min(r, kap_left[d[\"pool\"]][c])\n            r = min(r, src_left.setdefault",
    "            r = np.inf\n            r = min(r, src_left.setdefault",
), (
    "            if self.dtau[s] > 0:\n                r = min(r, fleet_left.get(d[\"pool\"], np.inf) / self.dtau[s])\n            return max(r, 0.0)",
    "            return max(r, 0.0)",
)])
# no position accounting of cargo in transit / waiting (only stock on hand)
make("abl_notransit", edits=[(
    "            position[(g, k)] = pl[\"s0\"] + sum(stock(f, k) for f in feeders) + sum(\n                transit[(n, k)] + q_in[(n, k)] for n in (g, *feeders)\n            )",
    "            position[(g, k)] = pl[\"s0\"] + sum(stock(f, k) for f in feeders)",
)])
# no safety stock and no threshold awareness at the valve is b_ss0 (already measured); reverse priority is z_rev
