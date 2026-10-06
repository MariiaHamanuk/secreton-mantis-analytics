p = "agents/chip_route/agent.py"
s = open(p).read()
s = s.replace('''    "pack_cover": 6.0,  # weeks of demand a market may hold: what plants still have is sent up to it (0: off)
''', '''    "pack_cover": 6.0,  # weeks of demand a market may hold: what plants still have is sent up to it (0: off)
    "pack_spill_all": False,  # the cover is also filled over lanes through straits
    "pack_spill_idle": False,  # the cover is filled even when no market needs anything
''')
s = s.replace('''    "w_u_min": 0.02,  # a fab with a smaller share gets no wafers
''', '''    "w_u_min": 0.02,  # a fab with a smaller share gets no wafers
    "w_u_pow": 1.0,  # the stock is scaled by usefulness to this power
''')
old = '''        if not any(need[m] > EPS for m in markets):
            return
'''
new = '''        if not any(need[m] > EPS for m in markets) and not (PARAMS["pack_cover"] > 0 and PARAMS["pack_spill_idle"]):
            return
'''
assert old in s
s = s.replace(old, new)
old = '''            for s, a in arc.items():
                if c.strait[s]:
                    fl.cap[a] = 0.0  # but not over a strait lane; what it carries so far stays
'''
new = '''            for s, a in arc.items():
                if c.strait[s] and not PARAMS["pack_spill_all"]:
                    fl.cap[a] = 0.0  # but not over a strait lane; what it carries so far stays
'''
assert old in s
s = s.replace(old, new)
old = '''                target *= 0.0 if use[n] < PARAMS["w_u_min"] else use[n]'''
new = '''                target *= 0.0 if use[n] < PARAMS["w_u_min"] else use[n] ** PARAMS["w_u_pow"]'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("ok")
