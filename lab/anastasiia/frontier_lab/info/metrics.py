"""Stage 3: discrimination, concentration, calibration and decision value from the score histograms."""

import pickle
import sys

import numpy as np


def pav(pos, tot):
    """Isotonic (non-decreasing) frequency by bin: pooled adjacent violators on bins with cases."""
    idx = np.flatnonzero(tot > 0)
    blocks = []  # [pos, tot, first, last]
    for i in idx:
        blocks.append([pos[i], tot[i], i, i])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][3] = b[3]
    out = np.zeros(len(tot))
    for p, t, a, b in blocks:
        out[a : b + 1] = p / t
    return out


def top(pos, tot, q):
    """Share of the positives among the top ``q`` of the cases by score (ties shared evenly inside a bin)."""
    need, got = q * tot.sum(), 0.0
    for i in range(len(tot) - 1, -1, -1):
        if tot[i] <= 0:
            continue
        take = min(tot[i], need)
        got += pos[i] * take / tot[i]
        need -= take
        if need <= 0:
            break
    return got / max(pos.sum(), 1e-12)


def summary(h):
    tot, pos, sp = h
    neg = tot - pos
    n, npos, nneg = tot.sum(), pos.sum(), neg.sum()
    base = npos / n
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    auc = float((pos * (below + 0.5 * neg)).sum() / max(npos * nneg, 1e-12))
    Hc = 1 - np.cumsum(pos) / max(npos, 1e-12)  # hit rate above each bin
    Fc = 1 - np.cumsum(neg) / max(nneg, 1e-12)
    ks = float(np.max(np.concatenate([[0.0], Hc - Fc])))
    cal = pav(pos, tot)
    share_two = float((pos * cal).sum() / max(npos, 1e-12))  # E[p | event] of the calibrated score
    share_point = float(2 * (tot * np.clip(cal - 0.5, 0, None)).sum() / max(npos, 1e-12))
    bss = float((tot * (cal - base) ** 2).sum() / max(n * base * (1 - base), 1e-12))

    # the calibration of the raw score: predicted over observed, overall and in the top 1 % and 10 % of the cases
    def ratio(q):
        need, s_p, s_y = q * n, 0.0, 0.0
        for i in range(len(tot) - 1, -1, -1):
            if tot[i] <= 0:
                continue
            take = min(tot[i], need)
            s_p += sp[i] * take / tot[i]
            s_y += pos[i] * take / tot[i]
            need -= take
            if need <= 0:
                break
        return s_p / max(s_y, 1e-12), s_y / (q * n)

    r1, f1 = ratio(0.01)
    r10, f10 = ratio(0.10)
    alerts = {}
    for th in (0.05, 0.1, 0.2, 0.5):
        sel = cal >= th
        alerts[th] = (
            float(tot[sel].sum() / n),
            float(pos[sel].sum() / max(npos, 1e-12)),
            float(pos[sel].sum() / max(tot[sel].sum(), 1e-12)),
        )
    cover8 = float((pos * (1 - (1 - cal) ** 8)).sum() / max(npos, 1e-12))
    return dict(
        cover8=cover8,
        alerts=alerts,
        n=n,
        npos=npos,
        base=base,
        auc=auc,
        top10=top(pos, tot, 0.10),
        top1=top(pos, tot, 0.01),
        ks=ks,
        share_two=share_two,
        share_point=share_point,
        bss=bss,
        cal_all=sp.sum() / max(npos, 1e-12),
        cal_top1=r1,
        freq_top1=f1,
        cal_top10=r10,
        freq_top10=f10,
        pmax=float(cal.max()),
    )


def main(path):
    d = pickle.load(open(path, "rb"))
    print(f"# {d['task']}, {d['n']} episodes; elements at risk {d['risk']}")
    H = d["hist"]
    items = ("strait", "grid", "proh", "edge", "edge_fuel")
    vars_ = ("clim", "warn", "events", "threads", "filter", "latent", "oracle_id")
    for it in items:
        for h in (1, 4, 13):
            print(f"\n## {it}, horizon {h} weeks")
            print(
                "variant     base     AUROC  top10%  lift10  top1%  lift1   KS    E[p|ev]  point  BSS    pred/obs all top1 top10  freq_top1 pmax"  # noqa: E501
            )
            for v in vars_:
                m = summary(H[(it, v, h)])
                print(
                    f"{v:10s} {m['base']:.5f}  {m['auc']:.3f}  {m['top10']:.3f}  {m['top10'] / 0.1:5.2f}  {m['top1']:.3f}  {m['top1'] / 0.01:5.1f}  "  # noqa: E501
                    f"{m['ks']:.3f}  {m['share_two']:.4f}  {m['share_point']:.4f} {m['bss']:.4f}  {m['cal_all']:.2f} {m['cal_top1']:.2f} {m['cal_top10']:.2f}  {m['freq_top1']:.4f} {m['pmax']:.3f}"  # noqa: E501
                )
    print(
        "\n## alerts: calibrated chance at or above a threshold: share of cases / share of events caught / frequency inside"  # noqa: E501
    )
    for it in items:
        for h in (1, 4, 13):
            for v in ("events", "threads", "filter", "latent", "oracle_id"):
                m = summary(H[(it, v, h)])
                print(
                    f"{it:10s} h={h:2d} {v:10s} cases {int(m['n'])} "
                    + "  ".join(f">={th}: {a[0]:.5f} / {a[1]:.3f} / {a[2]:.3f}" for th, a in m["alerts"].items())
                )
    print(
        "\n## inside an element (score over the element's base rate): AUROC, top 10 %; and the chance that one / at least one of eight scenarios drawn from the predictor holds the event"  # noqa: E501
    )
    for it in items:
        for h in (1, 4, 13):
            row = [f"{it:10s} h={h:2d}"]
            for v in ("warn_rel", "events_rel", "filter_rel"):
                m = summary(H[(it, v, h)])
                row.append(f"{v}: {m['auc']:.3f} {m['top10']:.3f}")
            for v in ("clim", "filter", "latent"):
                m = summary(H[(it, v, h)])
                row.append(f"{v}: one {m['share_two']:.3f} eight {m['cover8']:.3f}")
            print("  ".join(row))
    st = d["stats"]
    names = [
        "tariff",
        "sanction",
        "outage",
        "closure",
        "conflict",
        "piracy",
        "energy",
        "weather(cluster)",
        "strike(cluster)",
        "weather",
        "strike",
        "edge-weeks new",
        "fuel edge-weeks new",
        "edge-weeks carried",
        "fuel edge-weeks carried",
        "events carried",
        "events new",
    ]
    print("\n## per episode, mean")
    for i, nme in enumerate(names):
        print(f"{nme:28s} {st[:, i].mean():10.3f}")
    th = d["threads"]
    if len(th):
        print("\n## threads: (type, decoy, dated, weeks to effect, prior, filter posterior, latent posterior, age)")
        for ty, name in ((1, "sanction"), (3, "closure")):
            for dated in (0, 1):
                sel = (th[:, 0] == ty) & (th[:, 2] == dated)
                if sel.sum() < 20:
                    continue
                t = th[sel]
                real = t[:, 1] == 0
                print(
                    f"{name} dated={dated}: thread-weeks {len(t)}, real share {real.mean():.3f}, prior {t[:, 4].mean():.3f}"  # noqa: E501
                )
                for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 13), (13, 1e9)):
                    b = (t[:, 3] > lo) & (t[:, 3] <= hi)
                    if b.sum() < 20:
                        continue
                    row = [f"  to effect ({lo},{hi}]: n {int(b.sum())}, real {real[b].mean():.3f}"]
                    for col, nm in ((5, "filter"), (6, "latent")):
                        s = t[b, col]
                        r = real[b]
                        if 0 < r.sum() < len(r):
                            from scipy.stats import rankdata

                            rk = rankdata(s)
                            auc = (rk[r].sum() - r.sum() * (r.sum() + 1) / 2) / (r.sum() * (len(r) - r.sum()))
                        else:
                            auc = float("nan")
                        row.append(f"{nm}: mean post real {s[r].mean():.3f} decoy {s[~r].mean():.3f} AUROC {auc:.3f}")
                    print("; ".join(row))


if __name__ == "__main__":
    main(sys.argv[1])
