p = "agents/chip_route/agent.py"
s = open(p).read()
if "w_assume_full" not in s:
    s = s.replace('''    "w_max": 1.0,  # never more than this many weeks of effective capacity
''', '''    "w_max": 1.0,  # never more than this many weeks of effective capacity
    "w_assume_full": False,  # plan as if this week's starts use all the wafers on hand (a power window may be opening)
''')
    old = '''            frac = 1.0 if avail_prev <= EPS else min(1.0, last / max(EPS, min(cap_eff, avail_prev)))
'''
    new = '''            frac = 1.0 if avail_prev <= EPS or PARAMS["w_assume_full"] else min(1.0, last / max(EPS, min(cap_eff, avail_prev)))
'''
    assert old in s
    s = s.replace(old, new)
    open(p, "w").write(s)
print("w_assume_full" in s)
