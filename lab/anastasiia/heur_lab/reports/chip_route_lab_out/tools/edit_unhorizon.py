p = "agents/chip_route/agent.py"
s = open(p).read()
old = '''            if PARAMS["w_useful"] == "horizon":  # the chips the fab's outlets can still sell, beyond those already made
                made = self.stock_of(c, n, self.fab[n]["out"]) + sum(b.get(self.fab[n]["out"], 0.0) for b in c.wip.get(n, {}).values())
                target = min(target, max(0.0, rate[n] * (self.T - first_sale + 1) - made))
            elif PARAMS["w_useful"]:  # a fab whose chips can hardly leave gets fewer wafers, so a smaller claim on power'''
new = '''            if PARAMS["w_useful"]:  # a fab whose chips can hardly leave gets fewer wafers, so a smaller claim on power'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("ok")
