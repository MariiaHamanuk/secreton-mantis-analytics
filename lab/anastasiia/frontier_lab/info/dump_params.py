import dataclasses

import numpy as np
from shockbench_flow.disruption import hawkes, regime
from shockbench_flow.hosting.tasks import task_generator
from shockbench_flow.omega import codes


def show(obj, indent=0, maxlen=600):
    pad = " " * indent
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if dataclasses.is_dataclass(v):
                print(f"{pad}{f.name}:")
                show(v, indent + 2)
            else:
                s = repr(v)
                print(f"{pad}{f.name}: {s[:maxlen]}")
    else:
        print(pad + repr(obj)[:maxlen])


for task in ("small", "full"):
    inst, p = task_generator(task)
    print(
        "=" * 30,
        task,
        "T",
        inst.T,
        "regions",
        inst.regions,
        "nchk",
        len(inst.chokepoints),
        "edges",
        len(inst.edges),
        "grids",
        len(inst.grids),
    )
    if task == "small":
        show(p)
    else:
        print("active", p.active_regions)
        print("baselines", p.hawkes.baselines)
    Pi = regime.weekly_conflict_matrix(p.regime.P_yr)
    print("Pi weekly\n", Pi)
    print("onset split", regime.onset_split(Pi))
    print("tension entry", regime.tension_entry(p.regime), "exit", 1 / p.regime.tension_spell)
    print("class mult", regime._class_multipliers(inst, p.regime), inst.region_class)
    G = hawkes.branching_matrix(inst, p)
    R = len(inst.regions)
    print("spr", hawkes.spectral_radius(G))
    print("G PP diag", np.round(np.diag(G[:R, :R]), 4))
    print("G PP col sums (children per P event)", np.round(G[:, :R].sum(0), 3))
    print("G MM col sums (children per M event)", np.round(G[:, R:].sum(0), 3))
    print(
        "chk adjacency",
        {inst.nodes[c].id: [inst.regions[m] for m in ms] for c, ms in inst.chokepoint_adjacency.items()},
    )
    print("dyads", p.regime.dyads)
    print("codes EVENT_TYPES", codes.EVENT_TYPES)
    print("CHANNELS", codes.CHANNELS, "TYPE_CHANNELS", codes.TYPE_CHANNELS)
    print(
        "norm c_X region",
        regime.normaliser(p.latent_region),
        "chk",
        regime.normaliser(p.latent_chokepoint),
        "dyad",
        regime.normaliser(p.latent_dyad),
    )
