p = "agents/chip_route/agent.py"
s = open(p).read()

# 1. usefulness -> flow rates + usefulness
old = '''    def usefulness(self, c):
        """fab node -> share of the fab's nominal capacity whose chips can still reach a market (0..1).

        A max-flow of the fab alone: fab -> plants (alive raw routes; a plant packages at most its throughput) -> markets
        (alive packaged routes; a market takes at most its mean forecast demand). Lanes that start on one edge share it.
        """
        out = {}
        for f in self.fabs:'''
new = '''    def usefulness(self, c):
        """fab node -> share of the fab's nominal capacity whose chips can still reach a market (0..1)."""
        return {f: min(1.0, phi / max(self.fab[f]["cap0"], EPS)) for f, phi in self.sale_rates(c).items()}

    def sale_rates(self, c):
        """fab node -> chips a week that can still go from the fab to markets.

        A max-flow of the fab alone: fab -> plants (alive raw routes; a plant packages at most its throughput) -> markets
        (alive packaged routes; a market takes at most its mean forecast demand). Lanes that start on one edge share it.
        """
        out = {}
        for f in self.fabs:'''
assert old in s
s = s.replace(old, new)
old = '''            out[f] = min(1.0, fl.run(0, 1) / max(cap0, EPS))
        return out'''
new = '''            out[f] = fl.run(0, 1)
        return out'''
assert old in s
s = s.replace(old, new)

# 2. wafers
old = '''        use = self.usefulness(c) if PARAMS["w_useful"] else {}
        for n in self.fabs:'''
new = '''        rate = self.sale_rates(c) if PARAMS["w_useful"] else {}
        for n in self.fabs:'''
assert old in s
s = s.replace(old, new)
old = '''            if PARAMS["endgame"]:
                # wafers sent now arrive, are started, mature, are shipped, packaged and shipped again before the horizon ends
                raw = [s for s in self.raw_slots[n] if c.cap[s] > EPS]
                if not raw:
                    continue
                lead_w = min(c.lead[s] for s in slots)
                lead_r = min(c.lead[s] for s in raw)
                tau_o = min(self.osat[int(self.dest[s])]["tau"] for s in raw)
                lead_p = min((c.lead[s] for p in self.pack_slots for s in self.pack_slots[p] if c.cap[s] > EPS), default=1)
                if c.t + lead_w + tau + 1 + lead_r + tau_o + 1 + lead_p > self.T:
                    continue
            target = min(PARAMS["w_weeks"] * cap_eff, PARAMS["w_fill"] * self.storage.get((n, inp), float("inf")))
            if PARAMS["w_useful"]:  # a fab whose chips can hardly leave gets fewer wafers, so a smaller claim on power
                target *= 0.0 if use[n] < PARAMS["w_u_min"] else use[n]'''
new = '''            raw = [s for s in self.raw_slots[n] if c.cap[s] > EPS]
            if not raw:
                continue
            # wafers sent now arrive, are started, mature, are shipped, packaged and shipped again: the week of the first sale
            lead_w = min(c.lead[s] for s in slots)
            lead_r = min(c.lead[s] for s in raw)
            tau_o = min(self.osat[int(self.dest[s])]["tau"] for s in raw)
            lead_p = min((c.lead[s] for p in self.pack_slots for s in self.pack_slots[p] if c.cap[s] > EPS), default=1)
            first_sale = c.t + lead_w + tau + 1 + lead_r + tau_o + 1 + lead_p
            if PARAMS["endgame"] and first_sale > self.T:
                continue
            target = min(PARAMS["w_weeks"] * cap_eff, PARAMS["w_fill"] * self.storage.get((n, inp), float("inf")))
            if PARAMS["w_useful"] == "horizon":  # the chips the fab's outlets can still sell, beyond those already made
                made = self.stock_of(c, n, self.fab[n]["out"]) + sum(b.get(self.fab[n]["out"], 0.0) for b in c.wip.get(n, {}).values())
                target = min(target, max(0.0, rate[n] * (self.T - first_sale + 1) - made))
            elif PARAMS["w_useful"]:  # a fab whose chips can hardly leave gets fewer wafers, so a smaller claim on power
                u = min(1.0, rate[n] / max(self.fab[n]["cap0"], EPS))
                target *= 0.0 if u < PARAMS["w_u_min"] else u'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("ok")
