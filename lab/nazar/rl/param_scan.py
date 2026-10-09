"""One-at-a-time scan of the rules' numbers: writes one agent folder per change, then compares them to the base.

    uv run python lab/nazar/rl/param_scan.py --out=outputs/nazar_scan                 # writes the folders
    uv run python hub/eval/compare.py --base=agents/anastasiia_rules_v2 --episodes=32 --entropy=555 --n_jobs=2 \
        outputs/nazar_scan/pull_all__flip outputs/nazar_scan/rate_cap__to1.1 ...      # a few folders per process

Every number of ``PARAMS`` (chip_part.py) and ``FUEL`` (fuel_part.py) of ``agents/anastasiia_rules_v2``: floats x0.8 and
x1.25 (kept at most 1 where they are shares), integers +-1, booleans flipped, three zeros tried at 0.1 / 0.5 and
``rate_cap`` at 1.1. 99 folders of the rules (about 130 KB each) with a ``params.json``. Result of the scan on Small
root 555 x32: hub/tried/heuristics.md, "Скан 99 одиночних зсувів".
"""

import ast
import json
import shutil
from pathlib import Path

import fire


ROOT = Path(__file__).resolve().parents[3]
SHARES = {
    "on_ratio",
    "run_band",
    "overflow_frac",
    "gate_on",
    "gate_cover",
    "bank_ratio",
    "grid_buffer",
    "thr_full",
    "w_fill",
}


def numbers(src: Path) -> dict:
    out = {}
    for part in ("chip_part", "fuel_part"):
        for node in ast.parse((src / f"{part}.py").read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("PARAMS", "FUEL")
            ):
                out.update(ast.literal_eval(node.value))
    return out


def variants(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k in ("trace", "raw_waits", "fuel"):
            continue
        if isinstance(v, bool):
            out[f"{k}__flip"] = {k: not v}
        elif isinstance(v, int):
            for d in (-1, 1):
                if v + d >= 1:
                    out[f"{k}__{'p' if d > 0 else 'm'}1"] = {k: v + d}
        elif isinstance(v, float):
            if v == 0.0:
                for t in (
                    (0.1, 0.5)
                    if k in ("thr_margin", "thr_floor", "cover_weeks")
                    else ((1.1,) if k == "rate_cap" else ())
                ):
                    out[f"{k}__to{t}"] = {k: t}
            else:
                for f, tag in ((0.8, "x08"), (1.25, "x125")):
                    nv = round(v * f, 4)
                    out[f"{k}__{tag}"] = {k: min(nv, 1.0) if k in SHARES else nv}
    return out


def main(out: str = "outputs/nazar_scan", base: str = "agents/anastasiia_rules_v2") -> None:
    src, dest = ROOT / base, ROOT / out
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    found = variants(numbers(src))
    for name, change in found.items():
        folder = dest / name
        folder.mkdir()
        for f in ("chip_part.py", "fuel_part.py", "strait_part.py", "agent.py"):
            shutil.copy(src / f, folder / f)
        (folder / "params.json").write_text(json.dumps(change))
    print(f"{len(found)} folders in {dest}")


if __name__ == "__main__":
    fire.Fire(main)
