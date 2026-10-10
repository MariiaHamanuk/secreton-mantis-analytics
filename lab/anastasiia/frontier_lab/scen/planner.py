"""K planners of the team's kind, each told one sampled future of the network, and a rule for the week's action.

``Scen`` holds one instance of the lab's agent as it is (``point``: the forecast "as observed" with the tracker's
median ends) and K instances that are each told the marks of one scenario of ``world.World`` in place of that
forecast. Every instance sees the same real observation every week and carries its own plan; one action is played.

``choose``:

- "saa": the sample average. The candidates are the first weeks of the point plan and of the K scenario plans, and
  the mean of the K (``mean``). Candidate i is replayed on the simulator copy under scenario j's network, followed by
  scenario j's own plan from its second week on; its regret there is that cost minus the cost of j's own first week.
  A candidate's score is its mean regret over the scenarios other than its own (the own scenario would always vote
  for it). The point plan stands unless another candidate's score is lower by ``margin`` (bn USD). With
  ``recourse`` the scenario's later weeks may answer the candidate: its program is solved again with the first week
  held (``_again``).
- "joint": the two-stage program, one first week for all the scenarios (``_joint``); with ``point_weight`` the point
  forecast is a block of it and every planner carries the solution's later weeks.
- "oracle": a measuring tool, no agent: the candidates are judged by a planner told the episode's own network.
- "medoid": the scenario plan whose first week is nearest the elementwise median of the K first weeks.
- "mean": the mean of the K first weeks (the tanker releases of the medoid).
- "expect": one planner told the mean of ``expect`` scenarios' marks (no judge).
- "point": the point plan (the identity gate: the lab agent itself).
"""

import time

import numpy as np


class Scen:
    def __init__(self, Agent, config, world, K: int = 8, choose: str = "saa", mean: bool = True, expect: int = 0,
                 margin: float = 0.0, recourse: bool = False, carry: bool = True, solver: str = "simplex",
                 point_weight: float = 0.0, lean: bool = False, groups: tuple = (), truth=None) -> None:
        self.world, self.K, self.choose, self.mean, self.expect, self.margin = world, K, choose, mean, expect, margin
        self.recourse, self.carry, self.solver, self.w0 = recourse, carry, solver, float(point_weight)
        # ``lean``: a scenario's planner solves its exact cell only (no hull after its first week, no search), and the
        # point plan's first week is judged by the replay alone, with no program solved again
        self.lean = lean
        self.core = __import__("sys").modules[Agent.__module__]._core  # the lab agent's plan_core
        self.point = Agent(config)
        self.scen = []
        for _ in range(K + (1 if expect else 0)):
            ag = Agent(config)
            ag.p["truth"] = list(groups)
            if lean:
                ag.p["hull_every"], ag.p["search"] = 10**6, 0
            self.scen.append(ag)
        self.names = ["point"] + [f"s{j}" for j in range(K)] + (["exp"] if expect else [])
        # "oracle", a measuring tool and no agent: one more planner is told the episode's own network for its window
        # and judges the candidates (the point plan's and the scenario plans' first weeks; its own is not one)
        self.seer = None
        if choose == "oracle":
            self.seer = Agent(config)
            self.seer.p["truth"] = list(groups)
            self.seer.truth = truth
        self.truth = truth  # the episode's own marks: only to check that a scenario's present is the observed one
        self.log, self.rows = [], []
        self.notes = {"rows": self.rows, "present_bad": 0, "judge_errors": 0, "chosen": {}, "resolved": [0, 0]}

    # ----- the scenarios' marks -----------------------------------------------------------------------------------
    def _expected(self, week: int):
        """The mean of ``expect`` scenarios' marks (a prohibition and a war-risk class by the majority)."""
        from types import SimpleNamespace

        all_marks = [self.world.marks(j, week) for j in range(self.expect)]
        out = {}
        for name in vars(all_marks[0]):
            stack = np.stack([np.asarray(getattr(m, name), dtype=float) for m in all_marks])
            mean = stack.mean(axis=0)
            if name == "prohibited":
                mean = mean >= 0.5
            elif name == "wr_class":
                mean = np.rint(np.median(stack, axis=0)).astype(np.int8)
            out[name] = mean
        return SimpleNamespace(**out), all_marks

    # ----- the week ---------------------------------------------------------------------------------------------------
    def act(self, observation):
        t0 = time.process_time()
        week = int(np.asarray(observation["week"]).ravel()[0])
        actions, made = [], []
        marks = [self.world.marks(j, week) for j in range(self.K)] if not self.expect else None
        if self.expect:
            expected, marks = self._expected(week)
        if self.truth is not None and marks:
            from world import present_bad

            self.notes["present_bad"] += sum(bool(present_bad(m, self.truth, week)) for m in marks)
        for i, ag in enumerate([self.point] + self.scen):
            if i > 0:
                ag.truth = expected if (self.expect and i == self.K + 1) else marks[i - 1]
            before = getattr(ag, "last", None)
            actions.append(ag.act(observation))
            last = getattr(ag, "last", None)
            made.append(None if last is None or last is before else (last[0], ag.acts, last))
        row, action, name = {"week": week}, actions[0], "point"
        try:
            if self.seer is not None:
                before = getattr(self.seer, "last", None)
                self.seer.act(observation)
                last = getattr(self.seer, "last", None)
                if last is not None and last is not before and self.seer.acts is not None:
                    action, name = self._oracle(actions, (last[0], self.seer.acts, last), row)
            elif self.choose == "expect":
                action, name = actions[self.K + 1], "exp"
            elif self.choose == "joint" and self.K:
                action, name = self._joint(actions, made, row)
            elif self.choose != "point" and self.K:
                cands = list(actions)
                pick = self._choose(cands, made, row)
                action, name = cands[pick], self.names[pick] if pick < len(self.names) else "mean"
        except Exception as error:  # the point plan stands
            self.notes["judge_errors"] += 1
            row["error"] = repr(error)[:200]
            action, name = actions[0], "point"
        base = np.asarray(actions[0]["flows"], dtype=float)
        row |= {"pick": name, "cpu": time.process_time() - t0,
                "dist": float(np.abs(np.asarray(action["flows"], dtype=float) - base).sum() / max(base.sum(), 1e-9))}
        self.rows.append(row)
        self.notes["chosen"][name] = self.notes["chosen"].get(name, 0) + 1
        self.log.append((row["cpu"], name))
        return action

    def _oracle(self, actions: list, seen: tuple, row: dict) -> tuple:
        """The candidate first week that is cheapest under the episode's own network, the told planner's later weeks
        answering it (``_again``); the point plan's unless another is cheaper by ``margin``."""
        ep, cont, last = seen
        Z = np.asarray(ep.marks.prohibited[0])
        wires = [self.point.model.wire(a, Z) for a in actions[: self.K + 1]]
        keys = [(tuple(sorted(w[0].items())), tuple(sorted(w[1].items())), tuple(sorted(w[2]))) for w in wires]
        cost, seen_keys = np.empty(len(wires)), {}
        for i, key in enumerate(keys):
            if key not in seen_keys:
                recs, J = ep.simulate(ep.clean([wires[i]] + list(cont)))
                seen_keys[key] = self._again(ep, last, recs, J)
            cost[i] = seen_keys[key]
        best = int(np.argmin(cost))
        gain = float(cost[0] - cost[best]) / 1e11
        row |= {"oracle_gain": gain, "oracle_best": best, "distinct": len(seen_keys),
                "oracle_spread": float(cost.max() - cost.min()) / 1e11}
        if gain <= self.margin:
            return actions[0], "point"
        return actions[best], self.names[best]

    def _again(self, ep, last: tuple, recs: list, J: int) -> int:
        """The cost of a first week under a scenario when the weeks after it may answer it: the scenario's program
        inside the regimes of the replay ``recs``, its first week held at what the replay executed (dispatches and
        tanker releases), solved once and played; the cheaper of that and the replay itself (``J``)."""
        _ep, _d, _tweak, bonus, anchor, price, _H = last
        inst, first = ep.inst, recs[0]
        if not hasattr(ep, "_first"):
            ep.actions(np.zeros(ep.N))  # builds its tables of columns
        held = [(ep.col("x", 1, *inst.action_slots[s]), float(first.executed.get(int(s), 0.0))) for s in range(len(inst.action_slots))]
        released: dict = {}
        for o, (c, k, e, _lane) in enumerate(inst.override_slots):
            released[(c, k, e)] = released.get((c, k, e), 0.0) + float(first.override_executed.get(o, 0.0))
        held += [(j, released[(c, k, e)]) for c, k, e, _o, j in ep._first if j is not None]
        mode, ref = ep.regimes(recs)
        C = ep.cell(mode, ref, anchor, price, bonus)
        for j, q in held:
            C.lb[j] = C.ub[j] = min(max(q, C.lb[j]), C.ub[j])
        sol = ep.solve(C, method="auto", time_limit=10.0, what="judge")
        self.notes["resolved"][0] += 1
        if sol["status"] != "Optimal":
            return J
        self.notes["resolved"][1] += 1
        return min(J, ep.simulate(ep.actions(sol["x"]))[1])

    def _joint(self, actions: list, made: list, row: dict) -> tuple:
        """The two-stage program: one first week for every scenario, each scenario's own later weeks, the mean cost.

        It starts from the point plan's first week. Scenario j's block is its own program inside the regimes of the
        replay "that first week, then j's plan"; the blocks share the first week's columns (dispatches and tanker
        releases), but for an edge, or a strait, whose first week differs between the scenarios (a cut that ends
        within the week in some of them): there every block asks what its own network carries and the action asks
        the largest of them, which the simulator clips for free. The solution is played under every scenario; it
        stands if its mean cost is lower by ``margin`` than the mean cost of the point plan's first week with the
        scenarios' later weeks answering it (``_again``). The scenarios' planners then carry the solution's later
        weeks. Returns (the action, its name)."""
        import scipy.sparse as sp

        K, point = self.K, self.point
        judges = [j for j in range(1, K + 1) if made[j] is not None and made[j][1] is not None]
        if len(judges) < (1 if self.w0 > 0 else 2) or made[0] is None:
            return actions[0], "point"
        Z = np.asarray(made[judges[0]][0].marks.prohibited[0])
        wire0 = point.model.wire(actions[0], Z)
        blocks, start = [], []
        # ``point_weight``: the point forecast is a block too, with that weight (the scenarios share the rest), and
        # the point planner carries the solution's later weeks like every other: the plan played is then one plan
        # from week to week, and with no difference between the scenarios it is the point plan itself
        owners = [self.scen[j - 1] for j in judges]
        if self.w0 > 0:
            if made[0][1] is None:
                return actions[0], "point"
            judges, owners = [0] + judges, [point] + owners
        weights = np.full(len(judges), 1.0 / len(judges))
        if self.w0 > 0:
            weights = np.r_[self.w0, np.full(len(judges) - 1, (1.0 - self.w0) / (len(judges) - 1))]
        for j in judges:
            ep, cont, last = made[j]
            _ep, _d, _tweak, bonus, anchor, price, _H = last
            recs, J = ep.simulate(ep.clean([wire0] + list(cont)))
            start.append(J if self.lean else self._again(ep, last, recs, J))
            mode, ref = ep.regimes(recs)
            C = ep.cell(mode, ref, anchor, price, bonus)
            A, lo, hi = ep.rows(C)
            blocks.append((ep, A, lo, hi, ep.obj if C.cost is None else ep.obj + C.cost, C.lb, C.ub))
        ep0 = blocks[0][0]
        N, inst = ep0.N, ep0.inst
        if any(b[0].N != N for b in blocks):
            return actions[0], "point"
        # the first week's columns every scenario shares: the edge's and the strait's first week the same in all
        u1 = np.stack([np.asarray(b[0].marks.u[0], dtype=float) for b in blocks])
        o1 = np.stack([np.asarray(b[0].marks.o[0], dtype=float) for b in blocks])
        k1 = np.stack([np.asarray(b[0].marks.prohibited[0]) for b in blocks])
        shared = []
        for key, col in ep0.tm.items():
            if key[0] != "x":
                continue
            e, k = key[1], key[2]
            tail = inst.edges[e].tail
            same = np.ptp(np.where(np.isfinite(u1[:, e]), u1[:, e], 0.0)) <= 1e-9 * max(1.0, float(np.nanmax(np.where(np.isfinite(u1[:, e]), u1[:, e], 0.0))))
            if same and tail in inst.chokepoint_ordinal:
                same = np.ptp(o1[:, inst.chokepoint_ordinal[tail]]) <= 1e-12
            if same and not np.ptp(k1[:, e, k].astype(int)):
                shared.append(col)
        B, n_rows = len(blocks), sum(b[1].shape[0] for b in blocks)
        r = np.arange(len(shared) * (B - 1))
        cols = np.concatenate([np.asarray(shared) + b * N for b in range(1, B)] + [np.tile(shared, B - 1)])
        link = sp.csc_matrix((np.concatenate([np.ones(len(r)), -np.ones(len(r))]), (np.concatenate([r, r]), cols)),
                             shape=(len(r), B * N))
        A = sp.vstack([sp.block_diag([b[1] for b in blocks], format="csc"), link], format="csc")
        lo = np.concatenate([b[2] for b in blocks] + [np.zeros(len(r))])
        hi = np.concatenate([b[3] for b in blocks] + [np.zeros(len(r))])
        cost = np.concatenate([w * b[4] for w, b in zip(weights, blocks)])
        lb, ub = np.concatenate([b[5] for b in blocks]), np.concatenate([b[6] for b in blocks])
        t0 = time.process_time()
        status, x = self._lp(A, lo, hi, cost, lb, ub)
        row |= {"joint": status, "joint_cpu": time.process_time() - t0, "joint_cols": B * N, "joint_rows": n_rows,
                "shared": len(shared)}
        if x is None:
            return actions[0], "point"
        plans = [b[0].actions(x[i * N : (i + 1) * N]) for i, b in enumerate(blocks)]
        flows, releases = {}, {}
        for plan in plans:  # the largest request of the scenarios (they differ only where the first week does)
            for slot, q in plan[0][0].items():
                flows[slot] = max(flows.get(slot, 0.0), q)
            for slot, q in plan[0][1].items():
                releases[slot] = max(releases.get(slot, 0.0), q)
        first = (flows, dict(sorted(releases.items())), plans[0][0][2])
        played = [b[0].simulate(b[0].clean([first] + list(plan[1:])))[1] for b, plan in zip(blocks, plans)]
        gain = float(np.dot(weights, np.asarray(start, dtype=float) - np.asarray(played, dtype=float))) / 1e11
        row |= {"joint_gain": gain, "start": [c / 1e11 for c in start], "played": [c / 1e11 for c in played]}
        if gain <= self.margin:
            return actions[0], "point"
        planned = point._arrays(first)
        raised = set()  # the model's own free requests above a cut, as its ``act`` makes them
        if point.watch is not None and point.p["watch_ask"]:
            raised = point._ask_early(planned)
        if point.p["ask_scale"] > 1.0:
            point._ask_scaled(planned, raised)
        if self.carry:
            for owner, plan in zip(owners, plans):
                owner.acts = list(plan[1:])
        return {"flows": planned["flows"], "override_qty": planned["override_qty"],
                "release_mode": planned["release_mode"]}, "joint"

    def _lp(self, A, lo, hi, cost, lb, ub, limit: float = 120.0) -> tuple:
        """HiGHS on one program: (status, x or None)."""
        hs, Highs = self.core.highs()
        inf = hs.kHighsInf
        n_row, n_col = A.shape
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = n_col, n_row
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lb, np.where(np.isinf(ub), inf, ub)
        lp.row_lower_, lp.row_upper_ = np.where(np.isinf(lo), -inf, lo), np.where(np.isinf(hi), inf, hi)
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
        lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data.astype(np.float64)
        h = Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("solver", self.solver)
        h.setOptionValue("time_limit", float(limit))
        h.passModel(lp)
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        return status, (np.asarray(h.getSolution().col_value, dtype=float) if status == "Optimal" else None)

    def _consensus(self, actions: list) -> tuple:
        """(the index of the medoid among ``actions``, the mean action with the medoid's tanker releases)."""
        F = np.stack([np.asarray(a["flows"], dtype=float) for a in actions])
        scale = np.maximum(F.max(axis=0), 1e-9)
        medoid = int(np.argmin((np.abs(F - np.median(F, axis=0)) / scale).sum(axis=1)))
        mean = dict(actions[medoid]) | {"flows": F.mean(axis=0)}
        return medoid, mean

    def _choose(self, cands: list, made: list, row: dict) -> int:
        """The index of the week's action in ``cands`` (the instances' actions; the mean is added at its end)."""
        K = self.K
        medoid, mean = self._consensus(cands[1 : K + 1])
        if self.choose == "medoid":
            return 1 + medoid
        if self.mean or self.choose == "mean":
            cands.append(mean)
        if self.choose == "mean":
            return len(cands) - 1
        judges = [j for j in range(1, K + 1) if made[j] is not None and made[j][1] is not None]
        if len(judges) < 2:
            return 0
        model = self.point.model
        Z = np.asarray(made[judges[0]][0].marks.prohibited[0])
        wires = [model.wire(a, Z) for a in cands]
        keys = [(tuple(sorted(w[0].items())), tuple(sorted(w[1].items())), tuple(sorted(w[2]))) for w in wires]
        first = {}
        for i, key in enumerate(keys):
            first.setdefault(key, i)
        distinct = sorted(set(first.values()))
        row["distinct"] = len(distinct)
        if len(distinct) == 1:
            return 0
        cost = np.full((len(cands), len(judges)), np.nan)
        for c, j in enumerate(judges):
            ep, cont, last = made[j]
            for i in distinct:
                acts = ep.clean([wires[i]] + list(cont))
                recs, J = ep.simulate(acts)
                cost[i, c] = self._again(ep, last, recs, J) if self.recourse else J
        for i, key in enumerate(keys):
            cost[i] = cost[first[key]]
        own = np.array([cost[j, c] for c, j in enumerate(judges)])
        regret = (cost - own[None, :]) / 1e11  # bn USD
        score = np.empty(len(cands))
        for i in range(len(cands)):
            others = [c for c, j in enumerate(judges) if j != i]
            score[i] = regret[i, others].mean() if others else np.inf
        best = int(np.argmin(score))
        pick = best if score[0] - score[best] > self.margin else 0
        row |= {"score_point": float(score[0]), "score_best": float(score[best]), "best": best,
                "regret_point": [float(x) for x in regret[0]], "judges": len(judges),
                "score_mean": float(score[-1]) if self.mean else None}
        return pick
