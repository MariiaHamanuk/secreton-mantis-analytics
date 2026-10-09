# `cand1`: найсильніша модель фази за гарантією часу

**Чому вона тут.** Це єдина модель треку, про яку відомо, що вона тримає **нуль тижнів naive у копії серверного
контейнера на обох мережах** (Small: тиждень 1 — 0.871 с, медіана 0.259, макс 0.871 за бюджету 2; Full: 1.146 /
1.581 / 3.033 за бюджету 4). Для порівняння `anastasiia_plan_hull` у тому самому контейнері мав 95 тижнів зі 104
понад бюджет, а `anastasiia_plan_hull2` на борді — 19 тижнів naive. Бал: Small 111 ×64 — 0.8705, Full 111 ×32 —
**0.8999** (під `--cpu_budget`, тобто тиждень понад бюджет уже зіграний naive), 0 фолбеків із 3 328.

Склала її третя сесія (`cand1` у `agents/` цієї лабораторії); тут лежить рівно те, чого немає більше нікде в git.

**Чому саме три файли.** Решта теки побайтно дорівнює `agents/anastasiia_plan_hull` (перевірено `cmp`:
`chip_part.py`, `fuel_part.py`, `lp_part.py`, `strait_part.py`, `sim_model.py`, `hybrid_agent.py`,
`grid_recovery.json`, `sbfv/`), тож копія пакета в git не потрібна.

**Чому не з `build.py`.** Із закоміченого джерела `lab/anastasiia/regime_lab/` ця модель **не збирається**:
`core.py` там не має `lim()` і досі тримає п'ять місць `time_limit=time_limit`, тобто баг «прохід коштує до
10 × `solve_seconds`», через який на борді й були 47 тижнів naive. `plan_core.py` тут розходиться з джерелом на
380 рядків, `agent.py` — на 141. **Виправлення годинника треба занести в джерело окремо**; поки це не зроблено,
єдина відтворювана копія моделі — ця.

Зібрати теку, яку можна грати:

```bash
cp -r agents/anastasiia_plan_hull /tmp/cand1
cp lab/anastasiia/next_lab/model_cand1/{agent.py,plan_core.py,regime.json} /tmp/cand1/
uv run sbf check /tmp/cand1 --task=small --docker
uv run sbf check /tmp/cand1 --task=full --docker
```

Налаштування (`regime.json` тут): `share` 0.55, `solve_seconds` 1.2, `skip_weeks` 1, `hull_every` 2,
`hull_only` "tail", `anchor_every` 4, `end_left` true. Чому саме так — `../COORD.md`, розділ про `cand1`:
`skip_weeks` 1 знімає тиждень 1 (у нього сервер зараховує `Agent(config)`, і це давало 4.171 с із 4);
`solve_seconds` не можна тримати спільною константою для двох мереж (0.9 коштувало −0.011 на Small);
`share` 0.75 недостатньо, бо годинник перевіряється перед розв'язком, а програвання після нього не бачить.

**Чого про неї не відомо:** формальної оцінки на root 222 немає, і в Formal Results її немає.
