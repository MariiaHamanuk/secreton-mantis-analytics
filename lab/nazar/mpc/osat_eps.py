"""One plant's packaged chip per episode: is the disposal concentrated in a few episodes (blocked outlets)?

uv run python lab/nazar/mpc/osat_eps.py lab/nazar/agents/nazar_rules_lpraw --plant=osat_my --chip=chip_le
"""

import sys
from pathlib import Path

import fire
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
    episodes: int = 20,
    n_jobs: int = 8,
) -> None:
    lay = layout(task, entropy)
    _n, _c, s, storage, outlets = next(p for p in lay["plants"] if p[0] == plant and p[1] == chip)
    ix = [i for i, _ in outlets]
    runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
    print(f"{Path(agent).name}: {plant} {chip}, storage {storage:,.0f}")
    print(
        f"  {'ep':>3}{'disposed':>12}{'weeks full':>11}{'mean stock':>12}{'outflow total':>15}{'weeks outflow=0':>16}"
    )
    for n, r in enumerate(runs):
        st, di = r["stock"][:, s], r["disposal"][:, s]
        out = r["sent"][:, ix].sum(1)
        print(
            f"  {n:>3}{di.sum():12,.0f}{(st >= 0.95 * storage).sum():11d}{st.mean():12,.0f}{out.sum():15,.0f}{(out <= 1e-9).sum():16d}"
        )


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
