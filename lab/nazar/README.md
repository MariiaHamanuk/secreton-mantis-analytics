# lab/nazar

Сесії Claude з Nazar (8–9 жовтня 2026): LP у гібриді, діагностика слабких місць, дрібні негативні результати.
Що з цього відтворилося і що ні — у `hub/tried/mpc.md` («LP у гібриді: де він виграє…»), `hub/tried/heuristics.md`
(«Де правила і гібрид слабкі…») і `hub/tried/rl.md` («ES по порогах…»).

| що | де |
| --- | --- |
| агенти сесії: `nazar_rules_lpraw` (правила `rules_v2` + сирі чипи з першого тижня LP `mpc_det`, горизонт 24, без цілих змінних; борд 0.8354) і `nazar_rules_lpraw_v2` (+ `rate_cap` 1.5; борд 0.8365), зонд `probe_agent` (правила + слоти LP за `swap.json`: `mpc_k`, `mpc_overrides`, `blend`, `combine`, `scale`, `scale_tail`, `tail_nodes`, `weeks`, `mix`), `rules_v2_dev` (хук порогів для ES) | **лише локально, у git не лежать** (решта `lab/nazar/agents/`); sha256 відправлених zip: `01efce7b…` і `551f7381…` |
| **`agents/h2_opt`** — `anastasiia_plan_hull2` з прискореннями, що **не змінюють жодної дії агента**: кеш хешу вмісту вікна (`content_digest`), кеш `sale_rates`, `__deepcopy__`, що ділить статичні таблиці правил, ранній вихід у `fleet_slack`, зв'язані методи в `Cell.row`. Дії побітово збігаються з оригіналом (Full 12 тижнів, Small 52 тижні); CPU −9…−10 % на Full, −1…−2 % на Small. Борд (сабміт 972071, без двох останніх патчів): 0.8863, 16 фолбеків; sha256 теки `01333343…`. Різниця до оригіналу: `patches/speedup_same_actions.diff` | `agents/h2_opt`, `patches/` |
| ES по порогах паливного рішення (потрібен `rules_v2_dev`) | `rl/es_gate.py` |
| скан чисел правил | `rl/param_scan.py` |
| діагностика (усе read-only, повторює `play` з `heur_lab2/tools/account.py`) | `mpc/`: `wafer_diag.py`, `wafer_trace.py`, `osat_diag.py`, `osat_eps.py`, `osat_time.py`, `doomed.py`, `raw_diff.py`, `raw_clip.py`, `free_lost.py`, `fuel_trace.py`, `worst.py`, `full_run.py` |

Скрипти `mpc/` і `rl/es_gate.py` приймають шлях до будь-якого агента (`mpc/raw_diff.py` і `mpc/osat_*` та `wafer_*`
прямо посилаються на теки з `lab/nazar/agents/`: підстав свої).

`full_run.py` грає агента й базу партіями на тих самих епізодах з лічильником CPU і друкує накопичений бал після
кожної партії; `--resume` продовжує зупинений прогін. Приклад: `uv run python lab/nazar/mpc/full_run.py
lab/nazar/agents/nazar_rules_lpraw_v2 --task=full --episodes=400 --batch=8`.

Усі проби запускались у WSL (Linux); на Windows пакет не працює (`fcntl`).
