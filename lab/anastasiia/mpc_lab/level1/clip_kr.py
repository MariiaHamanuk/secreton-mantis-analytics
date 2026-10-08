"""KR gas slots in the hub's trajectory: requested vs executed, weeks 14-39; Qatar source stock; edge loads."""

import sys

import numpy as np

sys.path.insert(0, "lab/anastasiia/mpc_lab/level1")
from probeB import kr_slots, setup  # noqa: E402

LSF = setup()
for n in [int(x) for x in sys.argv[1].split(",")]:
    d, R, rep = LSF.verify_one(n)
    inst = R.inst
    ks = kr_slots(inst)
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    qa = next(i for i, st in enumerate(inst.stock_slots) if st.node == nid["src_qa_lng"])
    rs = R.records[14:40]
    print(f"ep {n}: Qatar lng stock end-of-week mean {np.mean([r.stock[qa] for r in rs]):.0f}, "
          f"weeks with < 500: {sum(r.stock[qa] < 500 for r in rs)}/26; lift {np.mean([r.lift[qa] for r in rs]):.0f}/wk")
    for s in ks:
        req = np.mean([r.requested.get(s, 0.0) for r in rs])
        exe = np.mean([r.executed.get(s, 0.0) for r in rs])
        print(f"   {LSF.slot_name(inst, s):70s} requested {req:7.0f} executed {exe:7.0f}")
    allq = [s for s, (e, k, l) in enumerate(inst.action_slots) if inst.nodes[inst.edges[e].tail].id == "src_qa_lng"]
    print(f"   all Qatar slots: requested {np.mean([sum(r.requested.get(s, 0) for s in allq) for r in rs]):.0f} "
          f"executed {np.mean([sum(r.executed.get(s, 0) for s in allq) for r in rs]):.0f}")
