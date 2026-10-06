p = "agents/chip_route/agent.py"
s = open(p).read()
old = '''        cap_left = c.u.copy()  # edge capacity not yet given to a dearer chip
        used = {}  # (plant, k) -> stock already assigned
        for k in self.pack_order:
            for s in self.pack_slots[k]:
                flows[s] = 0.0
'''
new = '''        cap_left = c.u.copy()  # edge capacity not yet given to a dearer chip
        used = {}  # (plant, k) -> stock already assigned
        for k in self.pack_order:
            c.cap_before[k] = cap_left.copy()  # what is left of the edges for this chip, for its plants' outlets
            for s in self.pack_slots[k]:
                flows[s] = 0.0
'''
assert old in s
s = s.replace(old, new)
old = '''        c = self._context(observation)
        if PARAMS["pack"]:
            self._pack_flows(c, flows)'''
new = '''        c = self._context(observation)
        c.cap_before = {}  # packaged chip -> edge capacity left for it after the dearer chips this week
        if PARAMS["pack"]:
            self._pack_flows(c, flows)'''
assert old in s
s = s.replace(old, new)
old = '''                row = self.demand_row.get((int(self.dest[s]), p))
                if row is not None and c.forecast[row].sum() > EPS:
                    total += c.cap[s]
        return min(total, self._throughput(c, o))'''
new = '''                row = self.demand_row.get((int(self.dest[s]), p))
                if row is not None and c.forecast[row].sum() > EPS:
                    cap = c.cap[s]
                    if PARAMS["outlet_net"] and p in c.cap_before:  # a cheaper chip shares its edges with the dearer ones
                        cap = min(cap, c.cap_before[p][int(self.edge[s])])
                    total += cap
        return min(total, self._throughput(c, o))'''
assert old in s
s = s.replace(old, new)
s = s.replace('''    "raw_waits": [2.0, 4.0, 8.0, 16.0],''', '''    "outlet_net": False,  # a plant's outlet for a cheaper chip is what the dearer chips leave of the shared edges this week
    "raw_waits": [2.0, 4.0, 8.0, 16.0],''')
open(p, "w").write(s)
print("ok")
