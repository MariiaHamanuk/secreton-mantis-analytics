p = "agents/chip_route/agent.py"
s = open(p).read()
old = '''        by_market = {}
        for s in self.pack_slots.get(p, ()):
            if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T:
                m = int(self.dest[s])
                row = self.demand_row.get((m, p))
                if row is not None and c.forecast[row].sum() > EPS:
                    cap = c.cap[s]
                    if PARAMS["outlet_net"] and p in c.cap_before:  # a cheaper chip shares its edges with the dearer ones
                        cap = min(cap, c.cap_before[p][int(self.edge[s])])
                    by_market[m] = by_market.get(m, 0.0) + cap
        total = 0.0
        for m, cap in by_market.items():
            if PARAMS["outlet_demand"]:  # a market takes no more than its demand, whatever the routes could carry
                cap = min(cap, float(c.forecast[self.demand_row[(m, p)]].mean()))
            total += cap
        return min(total, self._throughput(c, o))'''
new = '''        total = 0.0
        for s in self.pack_slots.get(p, ()):
            if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T:
                row = self.demand_row.get((int(self.dest[s]), p))
                if row is not None and c.forecast[row].sum() > EPS:
                    cap = c.cap[s]
                    if p in c.cap_before:  # a cheaper chip shares its edges with the dearer ones: it gets what they leave
                        cap = min(cap, c.cap_before[p][int(self.edge[s])])
                    total += cap
        return min(total, self._throughput(c, o))'''
assert old in s
s = s.replace(old, new)
s = s.replace('''    "outlet_net": True,  # a plant's outlet for a cheaper chip is what the dearer chips leave of the shared edges this week
    "outlet_demand": False,  # a plant's outlet into a market is at most that market's demand
''', '')
open(p, "w").write(s)
print("outlet_net" in s, "outlet_demand" in s)
