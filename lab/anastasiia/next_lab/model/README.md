# `anastasiia_plan_hull_x`: три файли понад `agents/anastasiia_plan_hull`

Решта теки моделі побайтно дорівнює `agents/anastasiia_plan_hull` (перевірено `cmp`: `chip_part.py`, `fuel_part.py`,
`lp_part.py`, `strait_part.py`, `sim_model.py`, `hybrid_agent.py`, `grid_recovery.json`, `sbfv/`), тому в git лежать
лише змінені файли, а не копія пакета.

Зібрати теку, яку можна грати:

```bash
cp -r agents/anastasiia_plan_hull /tmp/plan_hull_x
cp lab/anastasiia/next_lab/model/{agent.py,plan_core.py,regime.json} /tmp/plan_hull_x/
uv run python lab/anastasiia/regime_lab/play.py run /tmp/plan_hull_x --tag=x --task=small --entropy=111 --episodes=64
```

З усіма новими числами на нулі (`hull_tilt` 0, `commit` 0, `hull_fix` null, `end_left` false, `anchor_free` null,
`hull_rounds` 1, `share` 0, `solve_seconds` 30) тека дорівнює `anastasiia_plan_hull` **до цента** — ворота, які
тримались після кожної правки.

Що саме змінено, і що кожне число робить — у `../README.md`, розділ 9. Числа на наборах підбору там само.
