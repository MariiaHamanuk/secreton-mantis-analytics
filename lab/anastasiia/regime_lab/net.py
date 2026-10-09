"""A small attention network that names the whole weeks in place of the solve with the hull: fitting and export.

    uv run --with torch --with scikit-learn python lab/anastasiia/regime_lab/net.py fit small_555 small_555d1 full_555 full_555d1

The trees of ``shares.py`` decide grid by grid from sums over a grid's short weeks, and so lose the weeks themselves
and what the other grids of the same plan need. Here a plan's short (week, grid) are tokens read together
(``core.net_tokens``): every token attends to every other one, with a learnt leaning to the tokens of its own grid,
and gets the probability that the hull's rounding marks that week (and, as a side target, the share the hull asks of
it). Grids have no names in it, so one network serves Small and Full. The data are ``shares.py collect``'s records
(the hull's own states and the states a model's play meets); the last quarter of each set's episodes is held out.
``fit`` writes the weights as arrays into a copy of a trees' file (``--trees``), which ``core.share_model`` reads; the
agent then runs the network with numpy alone (``core.Net``).
"""

import math
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import core  # noqa: E402
import shares  # noqa: E402


def states(name: str, wide: bool = False) -> list:
    """One item a (episode, week): (episode, tokens (n, F), same (n, n), marked (n,), share (n,), grid of each token)."""
    meta, y, _last, X = shares.load(name)
    order = np.lexsort((meta[:, 2], meta[:, 3], meta[:, 1], meta[:, 0]))
    meta, y, X = meta[order], y[order], X[order]
    c = {n: core.SHARE_FEATURES.index(n) for n in ("rationed", "ration_need", "whole_prev")}
    key = meta[:, :2]
    cut = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1, len(y)]
    out = []
    for a, b in zip(cut[:-1], cut[1:]):
        rows, marked, share = {}, {}, {}
        g = meta[a:b, 3].astype(int)
        for gi in np.unique(g):
            r = np.flatnonzero(g == gi) + a
            x = X[r]
            got = shares._round(y[r], x[:, c["rationed"]] > 0.5, x[:, c["ration_need"]], x[:, c["whole_prev"]] > 0.5)
            rows[int(gi)] = [(int(meta[i, 2]), X[i]) for i in r]
            for j, i in enumerate(r):
                marked[(int(meta[i, 2]), int(gi))] = 1.0 if j in got else 0.0
                share[(int(meta[i, 2]), int(gi))] = float(y[i])
        keys, x, same = core.net_tokens(rows, float(meta[a, 5]), float(meta[a, 4]) if wide else None)
        out.append((int(meta[a, 0]), x, same, np.array([marked[k] for k in keys]), np.array([share[k] for k in keys]),
                    np.array([k[1] for k in keys])))
    return out


def fit(*names: str, hold: float = 0.25, width: int = 64, layers: int = 2, heads: int = 4, epochs: int = 40, lr: float = 2e-3,
        seed: int = 0, trees: str = "", out: str = "", threads: int = 2, drop: float = 0.0, decay: float = 1e-4,
        wide: bool = False) -> None:
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sets = {name: states(name, wide) for name in names}
    train, test = [], {}
    for name, items in sets.items():
        eps = sorted({it[0] for it in items})
        held = set(eps[int(len(eps) * (1 - hold)):])
        w = 1.0 / sum(1 for it in items if it[0] not in held)  # every set weighs the same, however many states it has
        train += [(it, w) for it in items if it[0] not in held]
        test[name] = [it for it in items if it[0] in held]
    allx = np.concatenate([it[1] for it, _w in train])
    mean, std = allx.mean(0), allx.std(0) + 1e-6
    F, N = allx.shape[1], max(len(it[1]) for items in sets.values() for it in items)

    def batch(items):
        B = len(items)
        x, same = np.zeros((B, N, F), dtype=np.float32), np.zeros((B, N, N), dtype=np.float32)
        z, y, pad = np.zeros((B, N), dtype=np.float32), np.zeros((B, N), dtype=np.float32), np.ones((B, N), dtype=bool)
        for i, it in enumerate(items):
            n = len(it[1])
            x[i, :n], same[i, :n, :n], z[i, :n], y[i, :n], pad[i, :n] = (it[1] - mean) / std, it[2], it[3], it[4], False
        return tuple(torch.from_numpy(a) for a in (x, same, z, y, pad))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
            self.f1, self.f2 = nn.Linear(width, 2 * width), nn.Linear(2 * width, width)
            self.same = nn.Parameter(torch.zeros(heads))
            self.drop = nn.Dropout(drop)

        def forward(self, h, same, pad):
            B, n, d = h.shape
            a = self.ln1(h)
            q, k, v = (lin(a).view(B, n, heads, d // heads).transpose(1, 2) for lin in (self.q, self.k, self.v))
            att = q @ k.transpose(-1, -2) / math.sqrt(d // heads) + self.same.view(1, heads, 1, 1) * same.unsqueeze(1)
            att = att.masked_fill(pad.view(B, 1, 1, n), -1e9).softmax(-1)
            h = h + self.drop(self.o((att @ v).transpose(1, 2).reshape(B, n, d)))
            return h + self.drop(self.f2(torch.relu(self.f1(self.ln2(h)))))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp, self.blocks = nn.Linear(F, width), nn.ModuleList(Layer() for _ in range(layers))
            self.ln, self.outp = nn.LayerNorm(width), nn.Linear(width, 2)

        def forward(self, x, same, pad):
            h = self.inp(x)
            for blk in self.blocks:
                h = blk(h, same, pad)
            return self.outp(self.ln(h))

    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    items, weights = [it for it, _w in train], np.array([w for _it, w in train])
    weights = weights / weights.mean()
    steps = epochs * math.ceil(len(items) / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    pos = float(sum((1 - it[3]).sum() for it in items) / max(1.0, sum(it[3].sum() for it in items)))
    def held_out() -> float:  # the mean over the sets of the average precision of "a week of this grid is marked"
        model.eval()
        got = []
        with torch.no_grad():
            for held in test.values():
                ps, zs = [], []
                for a in range(0, len(held), 512):
                    part = held[a:a + 512]
                    x, same, _z, _y, pad = batch(part)
                    o = torch.sigmoid(model(x, same, pad)[..., 0]).numpy()
                    for i, it in enumerate(part):
                        for gi in np.unique(it[5]):
                            m = np.flatnonzero(it[5] == gi)
                            ps.append(o[i, m].max())
                            zs.append(it[3][m].max())
                got.append(average_precision_score(zs, ps))
        model.train()
        return float(np.mean(got))

    best = (-1.0, None, 0)
    for epoch in range(epochs):
        order, total = rng.permutation(len(items)), 0.0
        for a in range(0, len(order), 256):
            idx = order[a:a + 256]
            x, same, z, y, pad = batch([items[i] for i in idx])
            w = torch.from_numpy(weights[idx].astype(np.float32)).view(-1, 1)
            o = model(x, same, pad)
            keep = (~pad).float() * w
            bce = nn.functional.binary_cross_entropy_with_logits(o[..., 0], z, reduction="none", pos_weight=torch.tensor(min(pos, 20.0) ** 0.5))
            loss = ((bce + (o[..., 1] - y) ** 2) * keep).sum() / keep.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 5 == 4 or epoch == epochs - 1:
            ap = held_out()
            if ap > best[0]:
                best = (ap, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1)
            print(f"epoch {epoch + 1}: loss {total / len(items):.4f}, held-out average precision of a grid {ap:.3f}", flush=True)
    model.load_state_dict(best[1])  # the weights of the best epoch on the held-out episodes
    model.eval()
    print(f"kept: epoch {best[2]}")

    arrays = {"layers": layers, "heads": heads, "mean": mean, "std": std}
    sd = {k: v.detach().numpy().astype(float) for k, v in model.state_dict().items()}
    arrays.update(in_w=sd["inp.weight"], in_b=sd["inp.bias"], ln_w=sd["ln.weight"], ln_b=sd["ln.bias"], out_w=sd["outp.weight"], out_b=sd["outp.bias"])
    for i in range(layers):
        for short, long in (("ln1", "ln1"), ("ln2", "ln2"), ("q", "q"), ("k", "k"), ("v", "v"), ("o", "o"), ("f1", "f1"), ("f2", "f2")):
            arrays[f"l{i}_{short}_w"], arrays[f"l{i}_{short}_b"] = sd[f"blocks.{i}.{long}.weight"], sd[f"blocks.{i}.{long}.bias"]
        arrays[f"l{i}_same"] = sd[f"blocks.{i}.same"]

    class Read:  # the agent's reading of the arrays, to score the held-out episodes with the code the agent runs
        files = [f"net_{k}" for k in arrays] + ["net_threshold"]

        def __getitem__(self, k):
            return 0.5 if k == "net_threshold" else arrays[k[4:]]

    net = core.Net(Read())
    scored = {name: [(it, 1.0 / (1.0 + np.exp(-net.out(it[1], it[2])[:, 0]))) for it in held] for name, held in test.items()}
    diff = max(float(np.abs(net.out(it[1], it[2]) - model(*[t for t in batch([it])][:2], batch([it])[4]).detach().numpy()[0, :len(it[1])]).max())
               for it in test[names[0]][:50])
    p_all = np.concatenate([p for rows in scored.values() for _it, p in rows])
    z_all = np.concatenate([it[3] for rows in scored.values() for it, _p in rows])
    threshold = float(np.sort(p_all)[::-1][max(int(z_all.sum()) - 1, 0)])  # as many weeks told as the hull marks, held out
    for name, rows in scored.items():
        p, z = np.concatenate([q for _it, q in rows]), np.concatenate([it[3] for it, _q in rows])
        any_p, any_z, same_first, near, both = [], [], 0, 0, 0
        for it, q in rows:
            for gi in np.unique(it[5]):
                m = it[5] == gi
                any_p.append(q[m].max())
                any_z.append(it[3][m].max())
                if it[3][m].any():
                    first, top = int(np.flatnonzero(it[3][m])[0]), int(np.argmax(q[m]))
                    both += 1
                    same_first += top == first
                    near += abs(top - first) <= 1
        said = p >= threshold
        print(f"{name}: held out {len(rows)} states, {len(z)} short weeks, marked {z.mean():.2%}; average precision of a week {average_precision_score(z, p):.3f}, "
              f"of a grid {average_precision_score(any_z, any_p):.3f}; where the hull marks a grid ({both}) the likeliest week is its first in {same_first / max(both, 1):.0%}, "
              f"within one in {near / max(both, 1):.0%}; at the threshold: of the hull's weeks told {z[said].sum() / max(z.sum(), 1):.0%}, of those told the hull's {z[said].mean() if said.any() else 0:.0%}")
    print(f"threshold {threshold:.3f} (as many weeks told as marked on the held-out episodes); numpy against torch on 50 states: {diff:.2e}")
    arrays["threshold"] = threshold
    base = dict(np.load(trees or HERE / "share_model.npz"))
    path = Path(out) if out else plan_out() / "kept" / "share_model_net.npz"
    np.savez_compressed(path, **base, **{f"net_{k}": np.asarray(v) for k, v in arrays.items()})
    print(f"{path}: {sum(np.asarray(v).size for v in arrays.values())} numbers of the network beside the trees of {trees or 'share_model.npz'}")


def thresholds(path: str, *names: str, hold: float = 0.25, found: float = 0.9) -> None:
    """Set the two thresholds of a fitted network from the held-out episodes of the named sets and write them into
    its file: a grid is asked a week when its likeliest week reaches the first (set where ``found`` of the grids the
    hull marks are found); its other weeks are asked when they reach the second (set where as many weeks are told
    as the hull marks). Numpy alone."""
    data = dict(np.load(path))

    class Read:
        files = list(data)

        def __getitem__(self, k):
            return data[k]

    net = core.Net(Read())
    tops, marked, p_all, z_all, same_first, near, both = [], [], [], [], 0, 0, 0
    for name in names:
        items = states(name, net.wide)
        eps = sorted({it[0] for it in items})
        held = set(eps[int(len(eps) * (1 - hold)):])
        for it in items:
            if it[0] not in held:
                continue
            p = 1.0 / (1.0 + np.exp(-net.out(it[1], it[2])[:, 0]))
            p_all.append(p)
            z_all.append(it[3])
            for gi in np.unique(it[5]):
                m = it[5] == gi
                tops.append(p[m].max())
                marked.append(bool(it[3][m].any()))
                if it[3][m].any():
                    first, top = int(np.flatnonzero(it[3][m])[0]), int(np.argmax(p[m]))
                    both, same_first, near = both + 1, same_first + (top == first), near + (abs(top - first) <= 1)
    tops, marked, p_all, z_all = np.array(tops), np.array(marked), np.concatenate(p_all), np.concatenate(z_all)
    first = float(np.sort(tops[marked])[int((1 - found) * marked.sum())])
    more = float(np.sort(p_all)[::-1][max(int(z_all.sum()) - 1, 0)])
    flagged = tops >= first
    data["net_threshold"], data["net_more"] = np.array(first), np.array(max(more, first))
    np.savez_compressed(path, **data)
    print(f"{path}: a grid is asked from {first:.3f}: finds {found:.0%} of the grids the hull marks held out, flags {flagged.mean():.0%} of all, "
          f"{marked[flagged].mean():.0%} of the flagged are the hull's (the hull marks {marked.mean():.1%}); its likeliest week is the hull's first in "
          f"{same_first / both:.0%}, within one in {near / both:.0%}; more weeks from {max(more, first):.3f}")


def plan_out() -> Path:
    import plan

    return plan.OUT


if __name__ == "__main__":
    fire.Fire({"fit": fit, "thresholds": thresholds})
