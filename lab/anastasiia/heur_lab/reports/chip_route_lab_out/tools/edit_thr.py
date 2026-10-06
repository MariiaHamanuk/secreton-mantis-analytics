p = "agents/chip_route/agent.py"
s = open(p).read()
old = '''            idx, n = {}, 3
            for x in plants + markets:
                idx[x] = n
                n += 1
            gate_of = {}
            for group in (rs, ps):'''
new = '''            idx, n = {}, 3
            for x in plants + markets:
                idx[x] = n
                n += 1
            idx_out = {}  # a plant packages at most thr_eff a week: its raw chips enter ``idx``, its packaged leave ``idx_out``
            for o in plants:
                idx_out[o] = n
                n += 1
            gate_of = {}
            for group in (rs, ps):'''
assert old in s
s = s.replace(old, new)
old = '''            gate_in = {}
            for src_node, group in ((2, rs), (None, ps)):
                for s in group:
                    a = src_node if src_node is not None else idx[int(self.tail[s])]'''
new = '''            gate_in = {}
            for o in plants:
                fl.add(idx[o], idx_out[o], self._throughput(c, o))
            for src_node, group in ((2, rs), (None, ps)):
                for s in group:
                    a = src_node if src_node is not None else idx_out[int(self.tail[s])]'''
assert old in s
s = s.replace(old, new)
# helper
old = '''    # ------------------------------------------------------------------ what a fab's chips can still become'''
new = '''    def _throughput(self, c, o):
        """What plant ``o`` can package a week now (its effective throughput), inf if the observation does not say."""
        t = float(c.thr_eff[self.osat_ord[o]])
        return t if np.isfinite(t) else float("inf")

    # ------------------------------------------------------------------ what a fab's chips can still become'''
assert old in s
s = s.replace(old, new, 1)
# outlet min with throughput in _plant_state
old = '''        p = self.osat[o]["pack"][r]
        outlet = self._outlet(c, o, p)
        lead_out ='''
new = '''        p = self.osat[o]["pack"][r]
        outlet = self._outlet(c, o, p)
        if PARAMS["thr_limit"]:
            outlet = min(outlet, self._throughput(c, o))
        lead_out ='''
assert old in s
s = s.replace(old, new)
s = s.replace('''    "outlet_net": False,''', '''    "thr_limit": False,  # a plant's outlet is also capped by what it can package a week (its effective throughput)
    "outlet_net": False,''')
open(p, "w").write(s)
print("ok")
