"""One plant's packaged chip over time (4-week bins): stock, disposal, asked and executed outflow, market served.

uv run python lab/nazar/mpc/osat_time.py lab/nazar/agents/nazar_rules_lpraw --plant=osat_my --chip=chip_le
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "nazar" / "mpc"))
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402
from osat_diag import layout  # noqa: E402


def main(
    agent: str,
    plant: str = "osat_my",
    chip: str = "chip_le",
    task: str = "small",
    entropy: int = 444,
    episodes: int = 12,
    n_jobs: int = 8,
) -> None:
    lay = layout(task, entropy)
    name, _c, s, storage, outlets = next(p for p in lay["plants"] if p[0] == plant and p[1] == chip)
    ix = [i for i, _ in outlets]
    runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
    print(
        f"{Path(agent).name}: {plant} {chip}, storage {storage:,.0f}, weeks in 4-week bins, mean over {episodes} episodes"
    )
    print(f"  {'weeks':<8}{'stock end':>11}{'disposed':>10}{'asked':>10}{'executed':>10}")
    T = runs[0]["stock"].shape[0]
    for lo in range(0, T, 4):
        w = slice(lo, min(lo + 4, T))
        st = np.mean([r["stock"][w, s].mean() for r in runs])
        di = np.mean([r["disposal"][w, s].sum() for r in runs])
        a = np.mean([r["asked"][w][:, ix].sum() for r in runs])
        e = np.mean([r["sent"][w][:, ix].sum() for r in runs])
        print(f"  {lo + 1:>2}-{min(lo + 4, T):<5}{st:11,.0f}{di:10,.0f}{a:10,.0f}{e:10,.0f}")


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
