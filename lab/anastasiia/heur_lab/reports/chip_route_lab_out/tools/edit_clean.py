p = "agents/chip_route/agent.py"
s = open(p).read()
s = s.replace('''    "pack_spill_all": False,  # the cover is also filled over lanes through straits
    "pack_spill_idle": False,  # the cover is filled even when no market needs anything
''', '')
s = s.replace('''    "w_u_pow": 1.0,  # the stock is scaled by usefulness to this power
''', '')
s = s.replace('''        if not any(need[m] > EPS for m in markets) and not (PARAMS["pack_cover"] > 0 and PARAMS["pack_spill_idle"]):
            return
''', '''        if not any(need[m] > EPS for m in markets):
            return
''')
s = s.replace('''                if c.strait[s] and not PARAMS["pack_spill_all"]:
                    fl.cap[a] = 0.0  # but not over a strait lane; what it carries so far stays''', '''                if c.strait[s]:
                    fl.cap[a] = 0.0  # but not over a strait lane; what it carries so far stays''')
s = s.replace('''use[n] ** PARAMS["w_u_pow"]''', '''use[n]''')
open(p, "w").write(s)
print("pack_spill" in s, "w_u_pow" in s)
