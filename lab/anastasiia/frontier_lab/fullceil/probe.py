"""Size and time of one whole episode of Full as a cell program, and whether a kept play replays to its cost.

    uv run python lab/anastasiia/frontier_lab/fullceil/probe.py 2 --tags=tah0c_f,h3_f --solve
"""

import time

import common as K
import fire


def main(n: int, tags: str | tuple = "tah0c_f", solve: bool = False, entropy: int = 444) -> None:
    import core  # regime_lab's

    tags = list(tags.split(",") if isinstance(tags, str) else tags)
    t0 = time.process_time()
    ep = core.Episode.of("full", entropy, n)
    print(f"ep {n}: T {ep.T}, columns {ep.N} (program {ep.n0}), rows {ep.base.shape[0]}, slots {len(ep.inst.action_slots)}, "
          f"override slots {len(ep.inst.override_slots)}, built in {time.process_time() - t0:.1f} s CPU, peak {K.rss_mb():.0f} MB", flush=True)
    for tag in tags:
        rec = K.kept(tag)[n]
        acts = K.sent_actions(ep, rec["sent"])
        t0 = time.process_time()
        recs, J = ep.simulate(acts)
        print(f"  {tag}: kept {rec['J'] / K.BN:.2f} bn, replayed without tanker releases {J / K.BN:.2f} bn "
              f"({(J - rec['J']) / K.BN:+.2f}), {time.process_time() - t0:.1f} s CPU", flush=True)
    if not solve:
        return
    t0 = time.process_time()
    mode, ref = ep.regimes(recs)
    t1 = time.process_time()
    C = ep.cell(mode, ref)
    t2 = time.process_time()
    print(f"  regimes {t1 - t0:.1f} s, cell {t2 - t1:.1f} s, peak {K.rss_mb():.0f} MB", flush=True)
    for method in ("ipm",):
        t0, w0 = time.process_time(), time.time()
        sol = ep.solve(C, method=method, time_limit=900.0)
        print(f"  solve {method}: {sol['status']}, claim {sol['J'] / 1e9:.2f} bn, {time.process_time() - t0:.1f} s CPU, "
              f"{time.time() - w0:.1f} s wall, peak {K.rss_mb():.0f} MB", flush=True)
        if sol["status"] == "Optimal":
            a2 = ep.actions(sol["x"])
            r2, J2 = ep.simulate(a2)
            print(f"  played {J2 / K.BN:.2f} bn (start {J / K.BN:.2f})", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
