"""The shares of whole weeks the solve with the hull asks for, as data for a model that would stand in for it.

    uv run python lab/anastasiia/regime_lab/shares.py collect outputs/regime_lab/agents/hull_rec --task=small --entropy=555 --episodes=48
    uv run python lab/anastasiia/regime_lab/shares.py table small_555 full_555

``collect`` plays episodes with an agent built with ``record_shares`` and keeps one row per (episode, week, week of
the window, grid) in ``outputs/regime_lab/shares/<task>_<entropy>.npz``: the share (``y``), what a model in its
place may know (``X``, ``core.SHARE_FEATURES``), and last week's share for the same calendar week (``last``, -1 when
there was none). No reference costs are needed, so nothing but the agent runs. ``table`` prints how the shares are
spread and how well the plainest stand-ins do: zero, last week's share, a linear fit.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import core  # noqa: E402
import plan  # noqa: E402


OUT = plan.OUT / "shares"
KEYS = ("episode", "week", "t", "grid", "window", "after")


def _episode(agent: str, task: str, entropy: int, n: int) -> list:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
    rows, before, week, done = [], {}, 0, False
    while not done:
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done, week = term or trunc, week + 1
        kept = ag.detail[-1].get("shares")
        now = {}
        if kept is not None:
            H, after, shares, feats, _marks = kept
            for (t, gi), y in shares.items():
                rows.append((n, week, t, gi, H, after, y, before.get((week + t - 1, gi), -1.0), *feats[(t, gi)]))
                now[(week + t - 1, gi)] = y
        before = now  # by calendar week, for next week's rows
    return rows


def collect(agent: str, task: str = "small", entropy: int = 555, episodes: int = 16, first: int = 0, n_jobs: int = 4,
            out: str = "") -> None:
    rows = Parallel(n_jobs=n_jobs)(delayed(_episode)(agent, task, entropy, n) for n in range(first, first + episodes))
    R = np.array([r for ep in rows for r in ep], dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (out or f"{task}_{entropy}.npz")
    if path.is_file():  # episodes kept before stay
        meta, y, last, X = load(path.stem)
        old = np.c_[meta, y, last, X]
        R = np.concatenate([old[~np.isin(old[:, 0], R[:, 0])], R])
    np.savez_compressed(path, rows=R, names=np.array(KEYS + ("y", "last") + core.SHARE_FEATURES), clean=1)
    print(f"{path}: {len(R)} rows, {len(set(R[:, 0]))} episodes")


def _clean(meta, y, X) -> int:
    """Rows collected before 8 October read "the week before (after) is whole" after the rounding had written its
    own weeks, so the neighbours of a marked week said so: the answer inside the features, which a model in the
    hull's place never sees. Here those values are set back (``X`` in place): the marked weeks are short weeks.
    Returns the number of values changed."""
    c = {n: core.SHARE_FEATURES.index(n) for n in ("rationed", "ration_need", "whole_prev", "whole_next")}
    order = np.lexsort((meta[:, 2], meta[:, 3], meta[:, 1], meta[:, 0]))
    key = meta[order][:, [0, 1, 3]]
    starts = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1, len(order)]
    changed = 0
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        t, x = meta[rows, 2], X[rows]
        for j in _round(y[rows], x[:, c["rationed"]] > 0.5, x[:, c["ration_need"]], x[:, c["whole_prev"]] > 0.5):
            if j > 0 and t[j - 1] == t[j] - 1 and X[rows[j - 1], c["whole_next"]] != 0.0:
                X[rows[j - 1], c["whole_next"]] = 0.0
                changed += 1
            if j + 1 < len(rows) and t[j + 1] == t[j] + 1 and X[rows[j + 1], c["whole_prev"]] != 0.0:
                X[rows[j + 1], c["whole_prev"]] = 0.0
                changed += 1
    return changed


def load(name: str):
    data = np.load(OUT / f"{name}.npz")
    R = data["rows"]
    k = len(KEYS)
    meta, y, last, X = R[:, :k], R[:, k], R[:, k + 1], R[:, k + 2 :].copy()
    if "clean" not in data.files:
        _clean(meta, y, X)
    if X.shape[1] < len(core.SHARE_FEATURES):  # kept before the last features were added: zeros, as "not there"
        X = np.c_[X, np.zeros((len(X), len(core.SHARE_FEATURES) - X.shape[1]))]
    return meta, y, last, X


def _fit(X, y):
    A = np.c_[np.ones(len(X)), X]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def table(*names: str, hold: float = 0.25) -> None:
    """For each set: the spread of the shares, and the error of zero, of last week's share and of a linear fit on
    the episodes held out (the last ``hold`` of them). MAE in shares of a week; "weeks" is the error of the sum of
    shares over a (week, grid), the number of whole weeks the rounding would mark."""
    for name in names:
        meta, y, last, X = load(name)
        eps = np.unique(meta[:, 0])
        test = np.isin(meta[:, 0], eps[int(len(eps) * (1 - hold)) :])
        w = _fit(X[~test], y[~test])
        pred = {"zero": np.zeros(test.sum()), "last week's": np.where(last[test] >= 0, last[test], 0.0),
                "linear": np.clip(np.c_[np.ones(test.sum()), X[test]] @ w, 0.0, 1.0)}
        print(f"{name}: {len(y)} rows, {len(eps)} episodes ({int(test.sum())} rows held out); share mean {y.mean():.3f}, "
              f"zero {(y < 1e-6).mean() * 100:.0f}%, one {(y > 1 - 1e-6).mean() * 100:.0f}%; a row had last week's share in {(last >= 0).mean() * 100:.0f}%")
        group = meta[test][:, [0, 1, 3]]  # episode, week, grid
        _u, inv = np.unique(group, axis=0, return_inverse=True)
        true_sum = np.bincount(inv.ravel(), weights=y[test])
        for label, p in pred.items():
            err = np.abs(p - y[test])
            sums = np.abs(np.bincount(inv.ravel(), weights=p) - true_sum)
            r2 = 1 - ((p - y[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()
            print(f"   {label:12s} MAE {err.mean():.4f}  R2 {r2:+.3f}  weeks: mean error {sums.mean():.3f}, within half a week {(sums < 0.5).mean() * 100:.0f}%")


def _plans(name: str):
    """One row per (episode, week, grid): the whole weeks the hull asked for there (the sum of its shares), last
    week's sum over the same calendar weeks, and the features of the grid's short weeks summed up: their count, and
    the mean, the first and the largest value of every feature."""
    meta, y, last, X = load(name)
    key = meta[:, [0, 1, 3]]
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(uniq)
    count = np.bincount(inv, minlength=n).astype(float)
    total = np.bincount(inv, weights=y, minlength=n)
    had = np.bincount(inv, weights=(last >= 0), minlength=n) > 0
    prev = np.where(had, np.bincount(inv, weights=np.where(last >= 0, last, 0.0), minlength=n), -1.0)
    order = np.lexsort((meta[:, 2], inv))  # by plan, then by the week of the window
    first = np.full(n, -1)
    first[inv[order][::-1]] = order[::-1]  # the row of the earliest short week of each plan
    mean = np.stack([np.bincount(inv, weights=X[:, j], minlength=n) / count for j in range(X.shape[1])], axis=1)
    top = np.full((n, X.shape[1]), -np.inf)
    np.maximum.at(top, inv, X)
    F = np.c_[count, meta[first, 4], meta[first, 5], meta[first, 2], mean, X[first], top]
    names = ["weeks_short", "window", "after", "first_week"] + [f"{s}_{c}" for s in ("mean", "first", "max") for c in core.SHARE_FEATURES]
    return uniq, total, prev, F, names


def sums(*names: str, hold: float = 0.25, trees: bool = True, last: bool = True) -> None:
    """How well the whole weeks a plan asks of a grid (the sum of the hull's shares) are told without the solve: by
    zero, by last week's sum, by a linear fit and by boosted trees (scikit-learn, for the look only; with last week's
    sum among the features), on the episodes held out. "marks" is the share of (plan, grid) whose rounded number of
    whole weeks is right. The last names are tested on the model fitted on the first: one network on another.
    ``--nolast``: the fits do not see last week's sum, as a model that never has the solve would not."""
    sets = {name: _plans(name) for name in names}
    first = names[0]
    uniq, y, prev, F, cols = sets[first]
    eps = np.unique(uniq[:, 0])
    test = np.isin(uniq[:, 0], eps[int(len(eps) * (1 - hold)) :])
    wide = lambda F, prev: np.c_[F, np.where(prev >= 0, prev, 0.0), (prev >= 0).astype(float)] if last else F  # noqa: E731
    Fp = wide(F, prev)
    models = {"linear": _fit(Fp[~test], y[~test])}
    if trees:
        from sklearn.ensemble import HistGradientBoostingRegressor

        models["trees"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Fp[~test], y[~test])

    def report(label, y, prev, Fp, rows):
        print(f"{label}: {int(rows.sum())} (plan, grid) rows; whole weeks asked: mean {y[rows].mean():.2f}, none {(y[rows] < 0.5).mean() * 100:.0f}%, three or more {(y[rows] >= 2.5).mean() * 100:.0f}%")
        pred = {"zero": np.zeros(int(rows.sum())), "last week's": np.where(prev[rows] >= 0, prev[rows], 0.0),
                "linear": np.clip(np.c_[np.ones(int(rows.sum())), Fp[rows]] @ models["linear"], 0.0, None)}
        if trees:
            pred["trees"] = np.clip(models["trees"].predict(Fp[rows]), 0.0, None)
        for name_, p in pred.items():
            err = np.abs(p - y[rows])
            r2 = 1 - ((p - y[rows]) ** 2).sum() / ((y[rows] - y[rows].mean()) ** 2).sum()
            print(f"   {name_:12s} MAE {err.mean():.3f} weeks  R2 {r2:+.3f}  marks right {(np.round(p) == np.round(y[rows])).mean() * 100:.0f}%  within half a week {(err < 0.5).mean() * 100:.0f}%")

    report(f"{first}, held-out episodes", y, prev, Fp, test)
    for other in names[1:]:
        u2, y2, prev2, F2, _ = sets[other]
        report(f"{other}, the model of {first}", y2, prev2, wide(F2, prev2), np.ones(len(y2), dtype=bool))
    if trees:
        from sklearn.inspection import permutation_importance

        imp = permutation_importance(models["trees"], Fp[test], y[test], n_repeats=3, random_state=0).importances_mean
        top = np.argsort(-imp)[:10]
        allcols = cols + (["last_sum", "had_last"] if last else [])
        print("   what the trees lean on: " + ", ".join(f"{allcols[i]} {imp[i]:.3f}" for i in top))


def _round(shares, rationed, need, whole_prev) -> list:
    """The weeks ``core._rounded`` would mark for one grid from its short weeks' shares, in the order of the weeks."""
    have, after, out = 0.0, False, []
    for i, share in enumerate(shares):
        after = after or bool(whole_prev[i])
        if not after and rationed[i] and need[i] > 0:
            price, ready = max(1.0, need[i]), have
        else:
            price, ready = 1.0, have + share
        have += share
        after = ready >= price - 1e-6
        if after:
            have -= price
            out.append(i)
    return out


def marks(name: str = "small_555", hold: float = 0.25) -> None:
    """The weeks marked from stand-ins for the hull's shares against the weeks marked from the shares themselves, on
    the episodes held out: last week's shares; and a total told by boosted trees from the state alone, spread over
    the grid's short weeks as its scarcest fuel's burn is spread (the shares the "burn" variant rounds)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    meta, y, last, X = load(name)
    uniq, total, _prev, F, _cols = _plans(name)
    eps = np.unique(uniq[:, 0])
    held = eps[int(len(eps) * (1 - hold)) :]
    test = np.isin(uniq[:, 0], held)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(F[~test], total[~test])
    told = dict(zip(map(tuple, uniq), np.clip(model.predict(F), 0.0, None)))
    c = {n: core.SHARE_FEATURES.index(n) for n in ("burn_min", "rationed", "ration_need", "whole_prev")}
    order = np.lexsort((meta[:, 2], meta[:, 3], meta[:, 1], meta[:, 0]))
    meta, y, last, X = meta[order], y[order], last[order], X[order]
    key = meta[:, [0, 1, 3]]
    starts = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    score = {n: [0, 0, 0, 0, 0] for n in ("last week's", "trees x burn", "the burn alone", "nothing")}  # same set, same count, same first, marked both, marked either
    groups = 0
    for a, b in zip(np.r_[0, starts], np.r_[starts, len(y)]):
        if meta[a, 0] not in held:
            continue
        groups += 1
        rat, need, prev = X[a:b, c["rationed"]] > 0.5, X[a:b, c["ration_need"]], X[a:b, c["whole_prev"]] > 0.5
        truth = _round(y[a:b], rat, need, prev)
        burn = X[a:b, c["burn_min"]]
        spread = burn / burn.sum() if burn.sum() > 0 else np.full(b - a, 1.0 / (b - a))
        guesses = {"last week's": np.where(last[a:b] >= 0, last[a:b], 0.0), "trees x burn": told[tuple(key[a])] * spread,
                   "the burn alone": burn, "nothing": np.zeros(b - a)}
        for n, g in guesses.items():
            got = _round(g, rat, need, prev)
            s = score[n]
            s[0] += got == truth
            s[1] += len(got) == len(truth)
            s[2] += (got[:1] == truth[:1])
            s[3] += len(set(got) & set(truth))
            s[4] += len(set(got) | set(truth))
    print(f"{name}: {groups} (plan, grid) on the held-out episodes; weeks marked from the stand-in against the hull's:")
    for n, s in score.items():
        print(f"   {n:15s} the same weeks {s[0] / groups * 100:.0f}%  the same number {s[1] / groups * 100:.0f}%  the same first week {s[2] / groups * 100:.0f}%  overlap of marked weeks {s[3] / max(1, s[4]) * 100:.0f}%")


def _table(name: str):
    """One row a (episode, week, grid), as the model reads it (``core.MODEL_FEATURES``): keys, features, whether the
    hull's shares mark a week there (the rounding of ``core._rounded``, redone on the rows), their sum, and where the
    first marked week falls (its week of the window, the grid's last short week, the short weeks' count, its place
    among them)."""
    meta, y, _last, X = load(name)
    order = np.lexsort((meta[:, 2], meta[:, 3], meta[:, 1], meta[:, 0]))
    meta, y, X = meta[order], y[order], X[order]
    key = meta[:, [0, 1, 3]]
    starts = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1, len(y)]
    c = {n: core.SHARE_FEATURES.index(n) for n in ("rationed", "ration_need", "whole_prev")}
    F, marked, total, where = [], [], [], []
    for a, b in zip(starts[:-1], starts[1:]):
        x, t = X[a:b], meta[a:b, 2]
        F.append(core.plan_features(t, x, float(meta[a, 4]), float(meta[a, 5])))
        got = _round(y[a:b], x[:, c["rationed"]] > 0.5, x[:, c["ration_need"]], x[:, c["whole_prev"]] > 0.5)
        marked.append(len(got) > 0)
        total.append(y[a:b].sum())
        # the first marked week (0 when none), the grid's last short week, their count, the marked one's place
        where.append((t[got[0]] if got else 0.0, t[-1], float(b - a), float(got[0]) if got else -1.0))
    keys = key[starts[:-1]]
    state = np.r_[0, np.flatnonzero(np.any(keys[1:, :2] != keys[:-1, :2], axis=1)) + 1, len(keys)]
    for a, b in zip(state[:-1], state[1:]):  # the grids of one plan, each among the others (``core.with_rivals``)
        wide = core.with_rivals({int(keys[i, 2]): F[i] for i in range(a, b)})
        for i in range(a, b):
            F[i] = wide[int(keys[i, 2])]
    return keys, np.array(F), np.array(marked), np.array(total), np.array(where)


def _arrays(est, name: str) -> dict:
    """A fitted scikit-learn boosting as arrays ``core.Trees`` reads: children are indices inside their own tree."""
    feature, threshold, left, right, value, start = [], [], [], [], [], []
    for (tree,) in est._predictors:
        nodes = tree.nodes
        leaf = nodes["is_leaf"].astype(bool)
        start.append(len(value))
        feature += list(np.where(leaf, 0, nodes["feature_idx"].astype(np.int64)))
        threshold += list(nodes["num_threshold"])
        left += list(np.where(leaf, -1, nodes["left"].astype(np.int64)))
        right += list(np.where(leaf, -1, nodes["right"].astype(np.int64)))
        value += list(nodes["value"])
    return {f"{name}_feature": np.array(feature, dtype=np.int32), f"{name}_threshold": np.array(threshold, dtype=float),
            f"{name}_left": np.array(left, dtype=np.int32), f"{name}_right": np.array(right, dtype=np.int32),
            f"{name}_value": np.array(value, dtype=float), f"{name}_start": np.array(start, dtype=np.int32),
            f"{name}_base": float(np.ravel(est._baseline_prediction)[0])}  # fmt: skip


def fit(*names: str, hold: float = 0.25, found: float = 0.9, out: str = "", when: bool = False, features: int = 0,
        write: bool = True) -> None:
    """Fit the trees that stand in for the solve with the hull on the named sets together and write them where the
    agent's build takes them (``share_model.npz`` beside this file): a classifier of "the hull marks a week of this
    grid", a regression of the whole weeks it asks for and, with ``when``, a regression of the week of the window
    its first whole week falls in (on the cases it marks; the agent then asks for that week and not for the one the
    start's own burn rounds into, ``core.told_weeks``). Scikit-learn fits (``uv run --with scikit-learn``); the
    agent reads the arrays with numpy. First the last ``hold`` of each set's episodes are held out: their scores are
    printed and the probability to act on is set where ``found`` of their marked cases are found; then the trees are
    fitted again on everything. ``features``: fit on the first so many features only; ``--nowrite``: the held-out
    scores alone, no file."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import average_precision_score, r2_score

    def trees(kind, X, y):
        est = HistGradientBoostingClassifier if kind == "mark" else HistGradientBoostingRegressor
        return est(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(X, y)

    sets = {name: _table(name) for name in names}
    if features:  # only the first so many of ``core.MODEL_FEATURES``: what the ones added later are worth
        sets = {name: (key, F[:, :features], m, tot, w) for name, (key, F, m, tot, w) in sets.items()}
    held = {}
    for name, (key, _F, _m, _t, _w) in sets.items():
        eps = np.unique(key[:, 0])
        held[name] = np.isin(key[:, 0], eps[int(len(eps) * (1 - hold)) :])
    X = np.concatenate([sets[n][1][~held[n]] for n in names])
    mark = trees("mark", X, np.concatenate([sets[n][2][~held[n]] for n in names]))
    total = trees("total", X, np.concatenate([sets[n][3][~held[n]] for n in names]))

    def placed(rows):  # for the cases marked among ``rows``: what the trees of the first whole week read, its week
        return [np.c_[sets[n][1], sets[n][4][:, 1:3]][rows[n] & sets[n][2]] for n in names], [sets[n][4][rows[n] & sets[n][2], 0] for n in names]

    if when:
        Xw, yw = placed({n: ~held[n] for n in names})
        first = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=0).fit(np.concatenate(Xw), np.concatenate(yw))
        for name in names:
            _key, F, m, _t, w = sets[name]
            rows = held[name] & m
            told = first.predict(np.c_[F, w[:, 1:3]][rows])
            off = np.abs(told - w[rows, 0])
            print(f"{name}: the first whole week on the {int(rows.sum())} marked cases held out: within half a week {np.mean(off < 0.5) * 100:.0f}%, within a week and a half "
                  f"{np.mean(off < 1.5) * 100:.0f}%, three or more weeks early {np.mean(told - w[rows, 0] <= -2.5) * 100:.0f}%, late {np.mean(told - w[rows, 0] >= 2.5) * 100:.0f}%")
    p_all, y_all = [], []
    for name in names:
        _key, F, m, t, _w = sets[name]
        p = mark.predict_proba(F[held[name]])[:, 1]
        p_all.append(p)
        y_all.append(m[held[name]])
        print(f"{name}: {len(m)} (plan, grid), marked {m.mean() * 100:.1f}%; held out: average precision {average_precision_score(m[held[name]], p):.3f} "
              f"(chance {m[held[name]].mean():.3f}), whole weeks R2 {r2_score(t[held[name]], total.predict(F[held[name]])):+.3f}")
    p_all, y_all = np.concatenate(p_all), np.concatenate(y_all)
    threshold = float(np.sort(p_all[y_all])[int((1 - found) * y_all.sum())])
    flagged = p_all >= threshold
    print(f"probability to act on: {threshold:.3f}: finds {found * 100:.0f}% of the marked cases held out, flags {flagged.mean() * 100:.0f}% of all, {y_all[flagged].mean() * 100:.0f}% of the flagged are marked")
    if not write:
        return
    X = np.concatenate([sets[n][1] for n in names])
    mark = trees("mark", X, np.concatenate([sets[n][2] for n in names]))
    total = trees("total", X, np.concatenate([sets[n][3] for n in names]))
    extra = {}
    if when:
        Xw, yw = placed({n: np.ones(len(sets[n][2]), dtype=bool) for n in names})
        Xw = np.concatenate(Xw)
        first = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=0).fit(Xw, np.concatenate(yw))
        extra = _arrays(first, "when")
    path = Path(out) if out else HERE / "share_model.npz"
    np.savez_compressed(path, threshold=threshold, features=np.array(core.MODEL_FEATURES[: X.shape[1]]), sets=np.array(names), **_arrays(mark, "mark"),
                        **_arrays(total, "total"), **extra)
    read = core.share_model(path)  # the agent's reading against scikit-learn's, on a sample
    rows = np.random.default_rng(0).choice(len(X), 200, replace=False)
    d_mark = max(abs(read[0].raw(X[i]) - float(mark.decision_function(X[i : i + 1])[0])) for i in rows)
    d_total = max(abs(read[1].raw(X[i]) - float(total.predict(X[i : i + 1])[0])) for i in rows)
    d_when = max(abs(read[3].raw(Xw[i]) - float(first.predict(Xw[i : i + 1])[0])) for i in range(min(200, len(Xw)))) if when else 0.0
    print(f"{path}: {len(X)} rows, {X.shape[1]} features, {path.stat().st_size / 1e3:.0f} KB; numpy against scikit-learn on 200 rows: {d_mark:.2e}, {d_total:.2e}, {d_when:.2e}")


if __name__ == "__main__":
    fire.Fire({"collect": collect, "table": table, "sums": sums, "marks": marks, "fit": fit})
