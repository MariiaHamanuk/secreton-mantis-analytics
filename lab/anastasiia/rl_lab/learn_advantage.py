"""Learn the search instead of running it: predict a candidate's exact advantage, then pick the best.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/learn_advantage.py \
        --dataset=outputs/rl_lab/cf_small --epochs=200

The search in `action_search.py` is strong but needs the simulator and the true future. This turns it into a
policy: the labels are the exact advantages `counterfactual.py` measured, so the learner is given a supervised
problem on an exact signal instead of a policy gradient through weeks of noise. At play time the model scores
every candidate from the observation alone — no rollouts, no foresight — and the best one is applied if its
predicted advantage is positive.

A candidate is a (coordinate, factor) pair, and a coordinate is a set of action slots. It is described to the
model by **what those slots look like**, never by its index: the slot features are pooled over the coordinate's
mask and joined with the week's global features and the factor. That keeps the model shape-agnostic, so weights
fitted on Small can score Full's coordinates too.

Reported before any episode is played:

- ``explained``: how much of the advantages' variance the model explains on held-out states.
- ``picked>0``: how often the model's best candidate really beats the rules.
- ``captured``: the mean **true** advantage of the model's pick over the mean true advantage of the best
  candidate. This is the fraction of the search's weekly edge the model keeps, and it is the number that says
  whether a policy is worth playing at all.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import nets  # noqa: E402
from action_search import FACTORS, groups_of  # noqa: E402


def config_for(task, entropy=111):
    import gymnasium as gym  # noqa: F401
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_gym import agent_config_from_reset

    from sbf_starter import env_id

    env = gym.make(env_id(task), regime="standard", entropy=entropy)
    obs, info = env.reset(seed=0)
    cfg = agent_config_from_reset(env, obs, info)
    env.close()
    return cfg


class Scorer(nn.Module):
    """One number per candidate: the advantage it is predicted to earn, as a share of the cost to go."""

    def __init__(self, sizes, width=96):
        super().__init__()
        self.slot_enc = nets.mlp([sizes["slot"] + sizes["global"], width, width])
        # the node the cargo is going to: whether a grid should get fuel this week depends on the grid's own state
        # (what it shed, what it generates), and that lives in the node table, not in the slots'
        self.node_enc = nets.mlp([sizes["node"], width // 2, width // 2])
        # a candidate: the pooled look of its slots (mean and max), its destination, the week, and the factor
        self.head = nn.Sequential(
            *nets.mlp([2 * width + width // 2 + sizes["global"] + 3, width, width]), nn.Linear(width, 1)
        )

    def forward(self, slot, glob, masks_idx, owner, factors, node, dest):
        """``masks_idx`` are the slots of every candidate's coordinate laid end to end, ``owner`` says whose.

        The pooling is a segment reduction rather than a loop over candidates: a state has about a hundred of them
        and the loop made a fit an hour's work.
        """
        g = glob.expand(slot.shape[0], -1)
        h = self.slot_enc(torch.cat([slot, g], dim=-1))
        n = factors.shape[0]
        rows = h[masks_idx]
        pooled = torch.cat([nets.segment_mean(rows, owner, n), nets.segment_max(rows, owner, n)], dim=-1)
        f = factors.unsqueeze(-1)
        extra = torch.cat([f, (f - 1.0).abs(), (f > 1.0).float()], dim=-1)
        gg = glob.expand(n, -1)
        where = self.node_enc(node)[dest]
        return self.head(torch.cat([pooled, where, gg, extra], dim=-1)).squeeze(-1)


def coordinate_destinations(cfg, grouping):
    """The node each coordinate sends to. Node 0 where a coordinate mixes destinations (the ``family`` grouping)."""
    static = cfg["static"]
    slots, edges, lanes = static["action_slots"], static["edges"], static["lanes"]
    route = [lanes["edges"][ln] if ln is not None else [e] for e, ln in zip(slots["edge"], slots["lane"])]
    dest = np.array([int(edges["head"][r[-1]]) for r in route])
    out = []
    for _, mask in groups_of(cfg, grouping):
        where = np.unique(dest[mask])
        out.append(int(where[0]) if where.size == 1 else 0)
    return out


def prepare(dataset, features, task, grouping, entropy=111):
    """The dataset as tensors: one record per state, with its candidates and their exact advantages."""
    cfg = config_for(task, entropy)
    groups = groups_of(cfg, grouping)
    masks = [np.flatnonzero(m) for _, m in groups]
    coord_dest = coordinate_destinations(cfg, grouping)
    out = []
    for row in dataset:
        if "features_row" not in row or not row["candidates"]:
            continue
        i = int(row["features_row"])
        slot = torch.tensor(features["slot"][i], dtype=torch.float32)
        node = torch.tensor(features["node"][i], dtype=torch.float32)
        glob = torch.tensor(features["global"][i], dtype=torch.float32).unsqueeze(0)
        plain = max(float(row["plain_cost_to_go"]), 1.0)
        idx, owner, factors, labels, dest = [], [], [], [], []
        for c in row["candidates"]:
            j = int(c["coordinate"])
            m = masks[j]
            if m.size == 0:
                continue
            idx.extend(m.tolist())
            owner.extend([len(factors)] * m.size)
            dest.append(coord_dest[j])
            factors.append(float(c["factor"]))
            labels.append((plain - float(c["cost_to_go"])) / plain)
        if len(factors) < 2:
            continue
        out.append(
            {
                "slot": slot,
                "node": node,
                "global": glob,
                "idx": torch.tensor(idx, dtype=torch.long),
                "owner": torch.tensor(owner, dtype=torch.long),
                "dest": torch.tensor(dest, dtype=torch.long),
                "factors": torch.tensor(factors, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.float32),
                "episode": row["episode"],
                "week": row["week"],
            }
        )
    return out, cfg, groups


def metrics(model, records):
    """Held-out quality, and the share of the search's weekly edge the model keeps."""
    picked_true, best_true, hits, err, var = [], [], 0, [], []
    with torch.no_grad():
        for r in records:
            pred = model(r["slot"], r["global"], r["idx"], r["owner"], r["factors"], r["node"], r["dest"])
            j = int(torch.argmax(pred))
            picked_true.append(float(r["labels"][j]))
            best_true.append(float(r["labels"].max()))
            hits += int(r["labels"][j] > 0)
            err.append(float(((pred - r["labels"]) ** 2).mean()))
            var.append(float(r["labels"].var(unbiased=False)))
    picked, best = np.array(picked_true), np.array(best_true)
    return {
        "states": len(records),
        "explained": round(1.0 - float(np.mean(err)) / max(float(np.mean(var)), 1e-12), 4),
        "picked_gt0": round(hits / max(len(records), 1), 4),
        "picked_mean": round(float(picked.mean()), 6),
        "best_mean": round(float(best.mean()), 6),
        "captured": round(float(picked.mean() / best.mean()) if best.mean() > 0 else 0.0, 4),
    }


def losses(pred, labels, ce_weight):
    """Mean squared error, plus a listwise term that optimises the pick rather than the prediction.

    The advantages are small (a thousandth to a few hundredths of the cost to go), so a model trained on squared
    error alone is well served by predicting zero everywhere. What matters at play time is only which candidate
    comes out on top, so the ranking is trained directly: a softmax over the state's candidates against a target
    that puts its mass on the one that truly won.
    """
    mse = nn.functional.mse_loss(pred, labels)
    if ce_weight <= 0:
        return mse, mse
    target = torch.zeros_like(labels)
    target[int(torch.argmax(labels))] = 1.0
    ce = -(target * torch.log_softmax(pred / max(float(labels.abs().max()), 1e-9), dim=0)).sum()
    return mse + ce_weight * ce, mse


def main(
    dataset,
    task="small",
    entropy=111,
    epochs=200,
    width=96,
    lr=1e-3,
    holdout=0.25,
    seed=0,
    ce_weight=0.01,
    weight_decay=0.0,
    out=None,
):
    folder = Path(dataset) if Path(dataset).is_absolute() else ROOT / dataset
    rows = json.loads((folder / "dataset.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    grouping = settings.get("grouping", "destination")
    torch.manual_seed(seed)
    if not (folder / "features.npz").is_file():
        raise SystemExit(f"{folder}/features.npz is missing: regenerate the dataset with --keep_features=True")
    features = np.load(folder / "features.npz")

    records, _, _ = prepare(rows, features, task, grouping, entropy)
    if not records:
        raise SystemExit("the dataset has no feature rows: regenerate it with --keep_features=True")
    sizes = {
        "slot": records[0]["slot"].shape[1],
        "node": records[0]["node"].shape[1],
        "global": records[0]["global"].shape[1],
    }
    # split by episode, never by state: two weeks of one episode are not independent
    episodes = sorted({r["episode"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_test = max(1, int(len(episodes) * holdout))
    test_eps = set(episodes[:n_test])
    train = [r for r in records if r["episode"] not in test_eps]
    test = [r for r in records if r["episode"] in test_eps]
    print(
        f"{len(records)} state(s) from {len(episodes)} episode(s), grouping {grouping}; "
        f"train {len(train)} / held out {len(test)} (episodes {sorted(test_eps)})",
        flush=True,
    )

    model = Scorer(sizes, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    order = np.arange(len(train))
    best = None
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        total = 0.0
        for i in order:
            r = train[i]
            pred = model(r["slot"], r["global"], r["idx"], r["owner"], r["factors"], r["node"], r["dest"])
            loss, mse = losses(pred, r["labels"], ce_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(mse)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            tr, te = metrics(model, train), metrics(model, test)
            print(
                f"  epoch {epoch:4d} loss {total / len(train):.3e} | train captured {tr['captured']:+.3f} "
                f"picked>0 {tr['picked_gt0']:.2f} | held out captured {te['captured']:+.3f} "
                f"picked>0 {te['picked_gt0']:.2f} explained {te['explained']:+.3f}",
                flush=True,
            )
            if best is None or te["captured"] > best[0]:
                best = (te["captured"], epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})

    out = Path(out) if out else folder / "scorer"
    out.mkdir(parents=True, exist_ok=True)
    if best is not None:
        model.load_state_dict(best[2])
    torch.save(model.state_dict(), out / "scorer.pt")
    final = metrics(model, test)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "grouping": grouping,
                "sizes": sizes,
                "width": width,
                "factors": FACTORS,
                "ce_weight": ce_weight,
                "weight_decay": weight_decay,
                "dataset": str(folder.relative_to(ROOT)),
                "best_epoch": best[1] if best else None,
                "held_out": final,
            },
            indent=2,
        )
    )
    print(f"\nheld out: {json.dumps(final)}")
    print(
        "captured is the share of the search's weekly edge the model keeps; below ~0.3 a policy is not worth "
        "playing, and the limit is what the search itself found"
    )
    print(f"written {out}", flush=True)
    return final


if __name__ == "__main__":
    import fire

    fire.Fire(main)
