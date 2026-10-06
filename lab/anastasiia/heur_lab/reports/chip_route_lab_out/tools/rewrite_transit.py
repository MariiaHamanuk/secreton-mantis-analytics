p = "agents/chip_route/agent.py"
s = open(p).read()
a = s.index("    def _transit(self, c):")
b = s.index("    def _wip(self, c):")
new = '''    def _queue_lots(self, obs):
        """[(strait, k, lane, next edge, arrival week, quantity)] of the cargo waiting at straits.

        Small and Full give the queue as one dense table (rows ``layout.lot_keys``, one column per week of arrival);
        Tiny gives a padded list of lots.
        """
        out = []
        qty = np.asarray(obs["queue_lots.qty"], dtype=float)
        seen = np.asarray(obs["queue_lots.qty.observed"]) == 1
        if self.lot_keys is None:
            lane, nxt = np.asarray(obs["queue_lots.lane"]), np.asarray(obs["queue_lots.next_edge"])
            chk, k = np.asarray(obs["queue_lots.chokepoint"]), np.asarray(obs["queue_lots.k"])
            week = np.asarray(obs["queue_lots.arrival_week"])
            for i in np.flatnonzero(seen):
                out.append((int(chk[i]), int(k[i]), int(lane[i]), int(nxt[i]), int(week[i]), float(qty[i])))
            return out
        for row in np.flatnonzero(seen.any(axis=1)):
            chk, k, ln, nxt = self.lot_keys[row]
            for col in np.flatnonzero(seen[row]):
                out.append((chk, k, ln, nxt, int(col) + 1, float(qty[row, col])))
        return out

    def _transit(self, c):
        """(node, k) -> [(arrival week at the node, quantity)] of cargo on its way there, queued cargo included.

        With ``queue_eta``, cargo that waits at a strait (or will join its queue) arrives when the strait's throughput has
        released what is ahead of it, oldest cargo first.
        """
        obs, edges, lanes, tau = c.obs, self.edges, self.lanes, c.tau
        out = {}
        lots = self._queue_lots(obs)
        queue_by_week = {}  # strait -> {arrival week: container cargo}
        for chk, k, _ln, _nxt, week, q in lots:
            if self.pool_ct[k]:
                book = queue_by_week.setdefault(chk, {})
                book[week] = book.get(week, 0.0) + q
        queue_now = {chk: sum(book.values()) for chk, book in queue_by_week.items()}
        live = np.asarray(obs["pipeline.qty.observed"]) == 1
        lane_seen = np.asarray(obs["pipeline.lane.observed"]) == 1
        e_a, k_a = np.asarray(obs["pipeline.edge"]), np.asarray(obs["pipeline.k"])
        l_a, q_a = np.asarray(obs["pipeline.lane"]), np.asarray(obs["pipeline.qty"])
        w_a = np.asarray(obs["pipeline.arrival_week"])
        for i in np.flatnonzero(live):
            e, k, q, aw = int(e_a[i]), int(k_a[i]), float(q_a[i]), int(w_a[i])
            if lane_seen[i]:
                ln = int(l_a[i])
                es = lanes["edges"][ln]
                pos = self.lane_pos[ln].get(e)
                rest = sum(tau[x] for x in es[pos + 1 :]) if pos is not None else 0
                dest = edges["head"][es[-1]]
            else:
                rest, dest = 0, edges["head"][e]
            head = edges["head"][e]
            if PARAMS["queue_eta"] and self.pool_ct[k] and head in self.chk_ord:  # it will join the queue at that strait
                kap = float(c.kappa[self.chk_ord[head]])
                left = queue_now.get(head, 0.0) - kap * max(0, aw - c.t)
                rest += int(np.ceil(max(0.0, left) / kap)) if kap > EPS else 10**6
            out.setdefault((dest, k), []).append((aw + rest, q))
        for chk, k, ln, nxt, week, q in lots:
            if ln < 0:
                continue
            es = lanes["edges"][ln]
            pos = self.lane_pos[ln].get(nxt)
            rest = sum(tau[x] for x in es[pos:]) if pos is not None else 1
            wait = 0
            if PARAMS["queue_eta"] and self.pool_ct[k]:
                kap = float(c.kappa[self.chk_ord[chk]])
                book = queue_by_week[chk]
                ahead = sum(v for w, v in book.items() if w < week) + 0.5 * book[week]
                wait = int(np.ceil(ahead / kap)) if kap > EPS else 10**6
            out.setdefault((edges["head"][es[-1]], k), []).append((c.t + wait + rest, q))
        return out

'''
s = s[:a] + new + s[b:]
s = s.replace('self.lot_keys = [tuple(int(x) for x in key) for key in layout["lot_keys"]]',
              'self.lot_keys = [tuple(int(x) for x in key) for key in layout["lot_keys"]] if layout.get("lot_keys") else None')
open(p, "w").write(s)
print("ok")
