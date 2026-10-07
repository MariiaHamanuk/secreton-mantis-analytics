# LSF: прийняті ходи паливного локального пошуку (planner_LSF.py)

Агент `anastasiia_hybrid_chiplp`, task small, root 444; три епізоди з найбільшим виграшем. Ходи в порядку прийняття; «тиждень» — тиждень відправки; кількість — у одиницях товару. Тип: valve — термінал → система (лаг 0), order — замовлення з джерела. Перевірено одним прогоном.

## Епізод 6: J 2613.29 → 2532.02 млрд USD (−81.27), прийнято 759 ходів

| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |
|---|---|---|---|---|---|---|
| 1 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т3: 8,849 → 13,273 | 0.000 |
| 2 | shift-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т2: 8,802 → 22,075; т3: 13,273 → 0 | 0.000 |
| 3 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т1: 1,679 → 12,716; т2: 22,075 → 11,038 | 0.000 |
| 4 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т29: 8,100 → 12,150 | 0.000 |
| 5 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т29: 5,520 → 8,280 | 0.000 |
| 6 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т40: 5,520 → 8,280 | 0.000 |
| 7 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т25: 5,520 → 8,280 | 0.000 |
| 8 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т51: 4,982 → 7,473 | 0.000 |
| 9 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т45: 4,803 → 7,204 | 0.000 |
| 10 | shift-2 | valve | lng | `tg.tb.term_eu.grid_eu` | т50: 3,090 → 7,872; т52: 4,782 → 0 | 0.000 |
| 11 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т22: 3,575 → 5,362 | 0.000 |
| 12 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 0 → 1,666; т36: 3,331 → 1,666 | 0.000 |
| 13 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т9: 3,182 → 4,773 | 0.000 |
| 14 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т14: 2,952 → 4,428 | 0.000 |
| 15 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т50: 2,668 → 0 | 0.001 |
| 16 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т51: 2,668 → 0 | 0.000 |
| 17 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т49: 2,451 → 0 | 0.001 |
| 18 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т17: 2,401 → 0; т18: 0 → 2,401 | 0.000 |
| 19 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т51: 2,317 → 3,499; т52: 2,366 → 1,183 | 0.000 |
| 20 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т48: 2,346 → 0 | 0.001 |
| 21 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т46: 2,191 → 1,096; т47: 187 → 1,282 | 0.001 |
| 22 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т50: 2,071 → 3,107 | 0.000 |
| 23 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т32: 2,036 → 0; т33: 0 → 2,036 | 0.000 |
| 24 | shift+1 | order | lng | `sea.tb.src_us_lng.term_eu` | т46: 1,985 → 0; т47: 139 → 2,124 | 2.929 |
| 25 | drop | order | lng | `sea.tb.src_us_lng.term_eu` | т49: 1,985 → 0 | 0.000 |
| 26 | drop | order | lng | `sea.tb.src_us_lng.term_eu` | т50: 1,985 → 0 | 0.000 |
| 27 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т48: 1,900 → 0 | 0.000 |
| 28 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т51: 1,871 → 2,806 | 0.000 |
| 29 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т50: 1,867 → 3,270; т51: 2,806 → 1,403 | 0.001 |
| 30 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т49: 1,810 → 2,716 | 0.001 |
| 31 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т48: 1,800 → 3,158; т49: 2,716 → 1,358 | 0.001 |
| 32 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т49: 1,358 → 2,037 | 0.001 |
| 33 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т39: 1,758 → 2,638 | 0.000 |
| 34 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т47: 1,744 → 0 | 0.002 |
| 35 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т44: 1,646 → 2,469 | 0.000 |
| 36 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т46: 1,626 → 2,439 | 0.000 |
| 37 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т46: 1,602 → 801 | 0.001 |
| 38 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т46: 801 → 401 | 0.001 |
| 39 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т45: 220 → 621; т46: 401 → 0 | 0.011 |
| 40 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 1,442 → 2,163 | 0.000 |
| 41 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 2,163 → 3,244 | 0.000 |
| 42 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т27: 385 → 3,629; т28: 3,244 → 0 | 0.000 |
| 43 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т38: 1,322 → 1,983 | 0.000 |
| 44 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т34: 1,394 → 2,091 | 0.000 |
| 45 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т33: 110 → 2,200; т34: 2,091 → 0 | 0.000 |
| 46 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т24: 1,372 → 2,058 | 0.000 |
| 47 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т24: 2,058 → 3,087 | 0.000 |
| 48 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т23: 146 → 3,233; т24: 3,087 → 0 | 0.000 |
| 49 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 1,350 → 2,025 | 0.000 |
| 50 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 2,025 → 3,038 | 0.000 |
| 51 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 3,038 → 4,556 | 0.000 |
| 52 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 1,350 → 2,025 | 0.000 |
| 53 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 162 → 837; т22: 1,350 → 675 | 0.000 |
| 54 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 675 → 0; т24: 0 → 675 | 0.000 |
| 55 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 215 → 1,565; т36: 1,350 → 0 | 0.000 |
| 56 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 1,349 → 2,024 | 0.000 |
| 57 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т36: 1,187 → 1,781 | 0.000 |
| 58 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т20: 1,047 → 1,570 | 0.773 |
| 59 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т4: 1,040 → 1,560 | 0.001 |
| 60 | shift-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т35: 0 → 1,039; т36: 1,039 → 0 | 0.000 |
| 61 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т33: 867 → 1,301 | 0.133 |
| 62 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т48: 867 → 0 | 0.001 |
| 63 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т49: 867 → 0 | 0.001 |
| 64 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 855 → 1,282 | 0.495 |
| 65 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 1,282 → 1,924 | 0.177 |
| 66 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 1,924 → 962; т2: 855 → 1,817 | 0.000 |
| 67 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т2: 1,817 → 908; т3: 855 → 1,763 | 0.000 |
| 68 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т3: 1,763 → 882; т4: 855 → 1,737 | 0.000 |
| 69 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т4: 1,737 → 868; т5: 855 → 1,723 | 0.000 |
| 70 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т5: 1,723 → 862; т6: 855 → 1,717 | 0.000 |
| 71 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т6: 1,717 → 858; т7: 855 → 1,713 | 0.000 |
| 72 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т7: 1,713 → 857; т8: 855 → 1,712 | 0.000 |
| 73 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т8: 1,712 → 856; т9: 855 → 1,711 | 0.000 |
| 74 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т9: 1,711 → 855; т10: 855 → 1,710 | 0.000 |
| 75 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т10: 1,710 → 855; т11: 855 → 1,710 | 0.000 |
| 76 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т11: 1,710 → 855; т12: 855 → 1,710 | 0.000 |
| 77 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т12: 1,710 → 855; т13: 855 → 1,710 | 0.000 |
| 78 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т13: 1,710 → 855; т14: 855 → 1,710 | 0.000 |
| 79 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т14: 1,710 → 855; т15: 855 → 1,710 | 0.000 |
| 80 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т15: 1,710 → 855; т16: 855 → 1,710 | 0.000 |
| 81 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т16: 1,710 → 855; т17: 855 → 1,710 | 0.000 |
| 82 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т47: 855 → 0 | 0.000 |
| 83 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 900 → 450; т4: 0 → 450 | 0.000 |
| 84 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 450 → 675 | 0.000 |
| 85 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 0 → 900; т37: 900 → 0 | 0.000 |
| 86 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т37: 0 → 900; т38: 900 → 0 | 0.000 |
| 87 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т41: 900 → 0; т43: 682 → 1,582 | 0.127 |
| 88 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т41: 0 → 900; т42: 900 → 0 | 0.000 |
| 89 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т33: 839 → 0 | 0.042 |
| 90 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т50: 836 → 1,254 | 0.000 |
| 91 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т44: 828 → 1,242 | 0.271 |
| 92 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т49: 828 → 1,242 | 0.004 |
| 93 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т43: 812 → 1,217 | 0.221 |
| 94 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т48: 812 → 1,217 | 0.001 |
| 95 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т47: 785 → 0 | 0.001 |
| 96 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т47: 780 → 1,169 | 0.000 |
| 97 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т45: 762 → 381 | 0.038 |
| 98 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т44: 672 → 1,053; т45: 381 → 0 | 0.035 |
| 99 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т44: 750 → 1,125 | 0.000 |
| 100 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т47: 790 → 1,185 | 0.029 |
| 101 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т26: 735 → 1,102 | 0.000 |
| 102 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т27: 735 → 1,102 | 0.000 |
| 103 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т41: 735 → 1,102 | 0.002 |
| 104 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т46: 735 → 1,102 | 0.000 |
| 105 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т20: 785 → 1,177 | 0.034 |
| 106 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т20: 1,177 → 1,766 | 0.006 |
| 107 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т20: 1,766 → 0; т21: 0 → 1,766 | 0.430 |
| 108 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т1: 712 → 1,069 | 1.457 |
| 109 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т1: 1,069 → 1,603 | 0.368 |
| 110 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т2: 712 → 356; т3: 712 → 1,069 | 0.000 |
| 111 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т2: 356 → 178; т3: 1,069 → 1,247 | 0.000 |
| 112 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т2: 178 → 89; т3: 1,247 → 1,336 | 0.000 |
| 113 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т4: 712 → 356; т5: 712 → 1,069 | 0.000 |
| 114 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т4: 356 → 178; т5: 1,069 → 1,247 | 0.000 |
| 115 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т4: 178 → 89; т5: 1,247 → 1,336 | 0.000 |
| 116 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т3: 706 → 1,059 | 2.038 |
| 117 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т2: 267 → 1,326; т3: 1,059 → 0 | 0.115 |
| 118 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т4: 706 → 0; т5: 706 → 1,412 | 0.000 |
| 119 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т5: 1,412 → 0; т6: 695 → 2,107 | 0.004 |
| 120 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 274 → 980; т34: 706 → 0 | 0.197 |
| 121 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т6: 2,107 → 1,054; т7: 473 → 1,527 | 0.000 |
| 122 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т6: 1,054 → 0; т8: 473 → 1,527 | 0.001 |
| 123 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т30: 691 → 1,036 | 0.061 |
| 124 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т10: 682 → 1,023 | 1.329 |
| 125 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т46: 718 → 1,078 | 0.000 |
| 126 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т1: 667 → 1,001 | 1.374 |
| 127 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т1: 1,001 → 1,501 | 2.061 |
| 128 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т1: 1,501 → 2,251 | 2.234 |
| 129 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 667 → 1,001 | 1.338 |
| 130 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 1,001 → 1,501 | 0.723 |
| 131 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 1,501 → 0; т3: 667 → 2,168 | 1.374 |
| 132 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т3: 2,168 → 3,252 | 3.235 |
| 133 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т3: 3,252 → 1,626; т4: 667 → 2,293 | 1.231 |
| 134 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т3: 1,626 → 2,439 | 3.349 |
| 135 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т4: 2,293 → 3,439 | 2.630 |
| 136 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т4: 3,439 → 1,720; т5: 667 → 2,387 | 0.723 |
| 137 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т4: 1,720 → 0; т6: 667 → 2,387 | 0.001 |
| 138 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т7: 667 → 0; т9: 667 → 1,334 | 0.000 |
| 139 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т8: 667 → 0; т9: 1,334 → 2,001 | 0.000 |
| 140 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т9: 2,001 → 0; т10: 667 → 2,668 | 0.001 |
| 141 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т10: 2,668 → 1,334; т11: 667 → 2,001 | 0.000 |
| 142 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т11: 2,001 → 1,001; т12: 667 → 1,668 | 0.000 |
| 143 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т11: 1,001 → 0; т13: 667 → 1,668 | 0.001 |
| 144 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т13: 1,668 → 834; т14: 667 → 1,501 | 0.000 |
| 145 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т13: 834 → 0; т15: 667 → 1,501 | 0.000 |
| 146 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т15: 1,501 → 750; т16: 667 → 1,417 | 0.000 |
| 147 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т15: 750 → 375; т16: 1,417 → 1,793 | 0.000 |
| 148 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т15: 375 → 188 | 0.002 |
| 149 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т34: 663 → 994 | 0.384 |
| 150 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т21: 706 → 0 | 0.066 |
| 151 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т32: 706 → 0 | 2.006 |
| 152 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т26: 650 → 975 | 0.049 |
| 153 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т26: 975 → 1,462 | 0.063 |
| 154 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т26: 1,462 → 2,193 | 0.085 |
| 155 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 640 → 960 | 0.033 |
| 156 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 960 → 1,439 | 0.045 |
| 157 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 1,439 → 2,159 | 0.059 |
| 158 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т42: 0 → 791; т43: 1,582 → 791 | 0.000 |
| 159 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т43: 791 → 396; т44: 597 → 993 | 0.156 |
| 160 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т43: 396 → 593 | 0.231 |
| 161 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т46: 617 → 0 | 0.001 |
| 162 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 0 → 612; т35: 612 → 0 | 0.128 |
| 163 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т24: 607 → 304 | 0.047 |
| 164 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т24: 304 → 455 | 0.019 |
| 165 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т43: 606 → 909 | 0.254 |
| 166 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т43: 909 → 455; т44: 280 → 734 | 0.271 |
| 167 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т43: 455 → 682 | 0.672 |
| 168 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т45: 645 → 968 | 0.229 |
| 169 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т21: 572 → 859 | 0.256 |
| 170 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т43: 542 → 271 | 0.119 |
| 171 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т43: 271 → 406 | 0.080 |
| 172 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т44: 493 → 247 | 0.440 |
| 173 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т43: 402 → 525; т44: 247 → 123 | 0.026 |
| 174 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т44: 123 → 62 | 0.031 |
| 175 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т1: 475 → 0; т2: 475 → 950 | 0.322 |
| 176 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т2: 950 → 0; т4: 475 → 1,425 | 0.001 |
| 177 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т3: 475 → 0; т4: 1,425 → 1,900 | 0.000 |
| 178 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 475 → 237; т6: 475 → 712 | 0.000 |
| 179 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 237 → 119; т6: 712 → 831 | 0.000 |
| 180 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 119 → 0; т7: 475 → 594 | 0.000 |
| 181 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т6: 831 → 416; т7: 594 → 1,009 | 0.000 |
| 182 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т6: 416 → 208; т7: 1,009 → 1,217 | 0.000 |
| 183 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т6: 208 → 104; т7: 1,217 → 1,321 | 0.000 |
| 184 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т7: 1,321 → 661; т8: 475 → 1,135 | 0.000 |
| 185 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т7: 661 → 330; т8: 1,135 → 1,466 | 0.000 |
| 186 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т7: 330 → 0; т9: 475 → 805 | 0.000 |
| 187 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т8: 1,466 → 0; т9: 805 → 2,271 | 0.000 |
| 188 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т9: 2,271 → 1,135; т10: 475 → 1,610 | 0.000 |
| 189 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т9: 1,135 → 0; т11: 475 → 1,610 | 0.001 |
| 190 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т10: 1,610 → 805; т11: 1,610 → 2,416 | 0.000 |
| 191 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т12: 475 → 0; т13: 475 → 950 | 0.000 |
| 192 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т13: 950 → 0; т15: 475 → 1,425 | 0.001 |
| 193 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т14: 475 → 0; т15: 1,425 → 1,900 | 0.000 |
| 194 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т16: 475 → 0; т17: 475 → 950 | 0.001 |
| 195 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т20: 475 → 712 | 0.088 |
| 196 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т20: 712 → 356; т21: 475 → 831 | 0.032 |
| 197 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т26: 475 → 0 | 0.005 |
| 198 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 475 → 712 | 0.002 |
| 199 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т43: 475 → 237 | 0.004 |
| 200 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т43: 237 → 356 | 0.011 |
| 201 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т43: 356 → 0; т45: 475 → 831 | 0.000 |
| 202 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т43: 0 → 475; т44: 475 → 0 | 0.139 |
| 203 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т7: 1,527 → 763; т8: 1,527 → 2,290 | 0.000 |
| 204 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т7: 763 → 382; т8: 2,290 → 2,672 | 0.000 |
| 205 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т7: 382 → 573 | 0.173 |
| 206 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т11: 473 → 0; т12: 473 → 946 | 0.000 |
| 207 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т12: 946 → 0; т13: 473 → 1,419 | 0.000 |
| 208 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т13: 1,419 → 710; т14: 473 → 1,183 | 0.000 |
| 209 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т15: 473 → 0; т16: 473 → 946 | 0.000 |
| 210 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т17: 473 → 237 | 0.149 |
| 211 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т5: 465 → 697 | 0.000 |
| 212 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 486 → 729 | 1.002 |
| 213 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 729 → 1,094 | 0.995 |
| 214 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т2: 486 → 729 | 0.344 |
| 215 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т3: 486 → 729 | 0.352 |
| 216 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т3: 729 → 364; т4: 486 → 850 | 0.000 |
| 217 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т3: 364 → 547 | 0.537 |
| 218 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 486 → 729 | 0.643 |
| 219 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 729 → 364; т6: 486 → 850 | 0.000 |
| 220 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 364 → 547 | 0.208 |
| 221 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 850 → 425; т7: 486 → 911 | 0.000 |
| 222 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 425 → 638 | 0.242 |
| 223 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 638 → 957 | 0.092 |
| 224 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т7: 911 → 456; т8: 486 → 942 | 0.000 |
| 225 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т11: 486 → 729 | 0.016 |
| 226 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т12: 486 → 0; т13: 486 → 972 | 0.000 |
| 227 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 972 → 486; т14: 486 → 972 | 0.000 |
| 228 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 486 → 243 | 0.001 |
| 229 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 243 → 122; т14: 972 → 1,094 | 0.000 |
| 230 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т14: 1,094 → 547; т15: 486 → 1,033 | 0.000 |
| 231 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т14: 547 → 0; т16: 486 → 1,033 | 0.000 |
| 232 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т16: 1,033 → 516; т17: 486 → 1,002 | 0.000 |
| 233 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т16: 516 → 0; т18: 342 → 859 | 0.038 |
| 234 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т21: 0 → 486; т22: 486 → 0 | 0.273 |
| 235 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т25: 486 → 729 | 0.173 |
| 236 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т24: 392 → 756; т25: 729 → 364 | 0.061 |
| 237 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т25: 364 → 0; т26: 486 → 850 | 0.000 |
| 238 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т26: 850 → 1,276 | 0.037 |
| 239 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т27: 486 → 0; т28: 486 → 972 | 0.000 |
| 240 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т28: 972 → 1,458 | 0.037 |
| 241 | x2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т28: 1,458 → 2,916 | 0.004 |
| 242 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т29: 486 → 0; т30: 486 → 972 | 0.000 |
| 243 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т30: 972 → 486; т31: 342 → 828 | 0.000 |
| 244 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т30: 486 → 243 | 0.001 |
| 245 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т30: 243 → 0; т31: 828 → 1,071 | 0.007 |
| 246 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т35: 486 → 243 | 0.000 |
| 247 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т44: 486 → 729 | 0.002 |
| 248 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т44: 729 → 1,094 | 0.030 |
| 249 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т45: 486 → 0; т46: 0 → 486 | 0.000 |
| 250 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т10: 473 → 0; т12: 0 → 473 | 0.000 |
| 251 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 466 → 698 | 0.148 |
| 252 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 698 → 1,047 | 0.170 |
| 253 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 1,047 → 1,571 | 0.088 |
| 254 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т26: 432 → 0 | 0.341 |
| 255 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т43: 463 → 695 | 0.102 |
| 256 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т43: 695 → 1,042 | 0.116 |
| 257 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т43: 1,042 → 1,563 | 0.133 |
| 258 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т10: 103 → 333; т11: 461 → 230 | 0.099 |
| 259 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т11: 230 → 0; т12: 0 → 230 | 0.000 |
| 260 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т46: 457 → 686 | 0.000 |
| 261 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т46: 686 → 1,028 | 0.000 |
| 262 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т46: 1,028 → 1,543 | 0.000 |
| 263 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т41: 418 → 627 | 0.343 |
| 264 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т23: 0 → 442; т24: 442 → 0 | 0.021 |
| 265 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т17: 432 → 648 | 0.000 |
| 266 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т17: 648 → 972 | 0.000 |
| 267 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т33: 432 → 648 | 0.024 |
| 268 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т32: 0 → 648; т33: 648 → 0 | 0.194 |
| 269 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т32: 0 → 216; т33: 432 → 216 | 0.012 |
| 270 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т43: 525 → 263; т44: 62 → 324 | 0.160 |
| 271 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т43: 263 → 0 | 0.020 |
| 272 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т44: 424 → 0 | 0.092 |
| 273 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т40: 383 → 574 | 0.301 |
| 274 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т40: 574 → 861 | 0.109 |
| 275 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т39: 149 → 580; т40: 861 → 431 | 0.273 |
| 276 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 407 → 818; т43: 411 → 0 | 0.069 |
| 277 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т29: 376 → 564 | 0.150 |
| 278 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т29: 564 → 846 | 0.207 |
| 279 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т28: 0 → 423; т29: 846 → 423 | 0.121 |
| 280 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т39: 334 → 734; т41: 400 → 0 | 0.011 |
| 281 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т24: 756 → 0; т25: 0 → 756 | 0.000 |
| 282 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 329 → 2,143; т27: 3,629 → 1,814 | 0.000 |
| 283 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т27: 1,814 → 0; т28: 0 → 1,814 | 0.000 |
| 284 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т38: 356 → 534 | 0.149 |
| 285 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т38: 534 → 800 | 0.078 |
| 286 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т37: 0 → 400; т38: 800 → 400 | 0.002 |
| 287 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т33: 0 → 369; т34: 369 → 0 | 0.014 |
| 288 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 333 → 499 | 0.100 |
| 289 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 499 → 748 | 0.090 |
| 290 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 748 → 1,122 | 0.010 |
| 291 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т17: 355 → 533 | 0.000 |
| 292 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т38: 57 → 385; т39: 328 → 0 | 0.017 |
| 293 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т9: 325 → 488 | 0.815 |
| 294 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т9: 488 → 0; т10: 0 → 488 | 0.000 |
| 295 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т11: 324 → 486 | 0.121 |
| 296 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т10: 1,023 → 1,509; т11: 486 → 0 | 0.037 |
| 297 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т23: 346 → 0; т24: 0 → 346 | 0.000 |
| 298 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 342 → 514 | 0.003 |
| 299 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 514 → 0; т34: 179 → 693 | 0.006 |
| 300 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т19: 342 → 0; т20: 90 → 433 | 0.000 |
| 301 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т41: 308 → 462 | 0.176 |
| 302 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т40: 147 → 378; т41: 462 → 231 | 0.074 |
| 303 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т41: 231 → 346 | 0.026 |
| 304 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т1: 329 → 494 | 0.000 |
| 305 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т25: 24 → 1,096; т26: 2,143 → 1,072 | 0.000 |
| 306 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 1,072 → 536; т27: 0 → 536 | 0.000 |
| 307 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 536 → 804 | 0.000 |
| 308 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т1: 300 → 450 | 0.618 |
| 309 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т1: 450 → 675 | 0.021 |
| 310 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т2: 300 → 450 | 0.084 |
| 311 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т4: 300 → 450 | 0.290 |
| 312 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т6: 300 → 450 | 0.007 |
| 313 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т8: 300 → 450 | 0.034 |
| 314 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т31: 320 → 480 | 0.181 |
| 315 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т44: 734 → 1,102 | 0.174 |
| 316 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т44: 1,102 → 551; т45: 245 → 796 | 0.493 |
| 317 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т46: 296 → 0 | 0.000 |
| 318 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т30: 271 → 407 | 0.063 |
| 319 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т30: 407 → 0; т32: 475 → 882 | 0.367 |
| 320 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т33: 114 → 399; т34: 285 → 0 | 0.001 |
| 321 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т6: 0 → 141; т7: 281 → 141 | 0.000 |
| 322 | half | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 141 → 70 | 0.121 |
| 323 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 70 → 105 | 0.145 |
| 324 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 258 → 387 | 0.031 |
| 325 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 387 → 581 | 0.014 |
| 326 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т38: 276 → 0; т39: 199 → 474 | 0.002 |
| 327 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т20: 252 → 126 | 0.111 |
| 328 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т4: 270 → 135; т5: 270 → 405 | 0.115 |
| 329 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т1: 234 → 351 | 0.164 |
| 330 | shift+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т1: 351 → 0; т2: 234 → 584 | 0.460 |
| 331 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т2: 584 → 876 | 1.204 |
| 332 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т2: 876 → 1,314 | 1.805 |
| 333 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т2: 1,314 → 1,972 | 0.803 |
| 334 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т3: 234 → 351 | 0.385 |
| 335 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т3: 351 → 526 | 0.481 |
| 336 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т3: 526 → 789 | 0.043 |
| 337 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т4: 234 → 351 | 0.480 |
| 338 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т4: 351 → 526 | 0.097 |
| 339 | shift+2 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т4: 526 → 0; т6: 234 → 759 | 0.000 |
| 340 | half | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т5: 234 → 117 | 0.001 |
| 341 | shift+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т5: 117 → 0; т6: 759 → 876 | 0.001 |
| 342 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т43: 243 → 0 | 0.529 |
| 343 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т45: 621 → 310; т46: 0 → 310 | 0.049 |
| 344 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 85 → 395; т45: 310 → 0 | 0.012 |
| 345 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т9: 233 → 349 | 0.218 |
| 346 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т9: 349 → 175; т10: 0 → 175 | 0.000 |
| 347 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т9: 175 → 87; т10: 175 → 262 | 0.038 |
| 348 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т11: 233 → 0; т12: 233 → 466 | 0.000 |
| 349 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т12: 466 → 0; т13: 233 → 698 | 0.000 |
| 350 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т12: 0 → 698; т13: 698 → 0 | 0.045 |
| 351 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т14: 233 → 0; т16: 233 → 466 | 0.000 |
| 352 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т15: 233 → 0; т16: 466 → 698 | 0.000 |
| 353 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т19: 233 → 466; т20: 233 → 0 | 0.002 |
| 354 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т27: 233 → 466; т29: 233 → 0 | 0.003 |
| 355 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т29: 0 → 233; т30: 233 → 0 | 0.035 |
| 356 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т30: 0 → 233; т31: 233 → 0 | 0.070 |
| 357 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т35: 216 → 0; т36: 719 → 935 | 0.001 |
| 358 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т25: 226 → 113 | 0.001 |
| 359 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т25: 113 → 170 | 0.010 |
| 360 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т25: 170 → 0; т27: 0 → 170 | 0.000 |
| 361 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т1: 224 → 0; т2: 270 → 494 | 0.039 |
| 362 | shift+2 | valve | crude | `tg.tb.term_kr.grid_kr` | т3: 220 → 0; т5: 112 → 332 | 0.000 |
| 363 | shift+2 | valve | crude | `tg.tb.term_kr.grid_kr` | т22: 220 → 0; т24: 177 → 397 | 0.000 |
| 364 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т44: 220 → 330 | 0.000 |
| 365 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т34: 0 → 782; т35: 1,565 → 782 | 0.000 |
| 366 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 782 → 1,174 | 0.000 |
| 367 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т25: 214 → 0; т27: 0 → 214 | 0.000 |
| 368 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т22: 208 → 0; т23: 145 → 353 | 0.001 |
| 369 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т35: 204 → 0; т36: 66 → 269 | 0.000 |
| 370 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 395 → 489; т44: 188 → 94 | 0.003 |
| 371 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т44: 94 → 141 | 0.017 |
| 372 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т44: 141 → 212 | 0.013 |
| 373 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т42: 201 → 0; т43: 211 → 412 | 0.031 |
| 374 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 200 → 396; т6: 196 → 0 | 0.023 |
| 375 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т37: 189 → 284 | 0.130 |
| 376 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т37: 284 → 0; т38: 270 → 554 | 0.000 |
| 377 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т40: 0 → 183; т41: 183 → 0 | 0.000 |
| 378 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т34: 693 → 346 | 0.001 |
| 379 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 0 → 346; т34: 346 → 0 | 0.021 |
| 380 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т26: 167 → 250 | 0.044 |
| 381 | half+1 | valve | crude | `tg.tb.term_tw.grid_tw` | т1: 163 → 81; т2: 108 → 189 | 0.012 |
| 382 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 837 → 418; т22: 0 → 418 | 0.000 |
| 383 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 418 → 628 | 0.000 |
| 384 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 628 → 942 | 0.000 |
| 385 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т19: 162 → 243 | 0.000 |
| 386 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т19: 243 → 364 | 0.000 |
| 387 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т39: 580 → 870 | 0.337 |
| 388 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т6: 0 → 159; т8: 159 → 0 | 0.002 |
| 389 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т22: 151 → 0; т23: 442 → 593 | 0.000 |
| 390 | half-1 | order | lng | `sea.tb.src_us_lng.term_eu` | т46: 0 → 1,062; т47: 2,124 → 1,062 | 0.728 |
| 391 | x1.5 | order | lng | `sea.tb.src_us_lng.term_eu` | т47: 1,062 → 1,593 | 2.129 |
| 392 | x1.5 | order | lng | `sea.tb.src_us_lng.term_eu` | т47: 1,593 → 2,389 | 1.477 |
| 393 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 418 → 2,035; т23: 3,233 → 1,617 | 0.000 |
| 394 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т23: 145 → 218 | 0.135 |
| 395 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т23: 218 → 0; т24: 99 → 317 | 0.000 |
| 396 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т46: 142 → 0 | 0.000 |
| 397 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 142 → 213 | 0.000 |
| 398 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 213 → 319 | 0.000 |
| 399 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 319 → 0 | 0.000 |
| 400 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т19: 140 → 0; т20: 0 → 140 | 0.000 |
| 401 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т35: 139 → 0; т36: 108 → 247 | 0.000 |
| 402 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т39: 372 → 495; т41: 123 → 0 | 0.023 |
| 403 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 124 → 187 | 0.095 |
| 404 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 187 → 0; т37: 0 → 187 | 0.000 |
| 405 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т44: 113 → 169 | 0.004 |
| 406 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т44: 169 → 253 | 0.010 |
| 407 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т29: 0 → 111; т30: 111 → 0 | 0.026 |
| 408 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т31: 111 → 0 | 0.107 |
| 409 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т18: 110 → 0; т19: 140 → 250 | 0.035 |
| 410 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т33: 2,200 → 1,100; т34: 782 → 1,883 | 0.000 |
| 411 | shift+2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т24: 109 → 0; т26: 0 → 109 | 0.000 |
| 412 | shift+2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т25: 109 → 0; т27: 0 → 109 | 0.000 |
| 413 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т35: 109 → 163 | 0.000 |
| 414 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т36: 247 → 124; т37: 44 → 167 | 0.000 |
| 415 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т36: 124 → 62; т37: 167 → 229 | 0.000 |
| 416 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т36: 62 → 93 | 0.059 |
| 417 | shift+1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т35: 106 → 0; т36: 0 → 106 | 0.000 |
| 418 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 98 → 49 | 0.000 |
| 419 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т44: 94 → 195; т45: 101 → 0 | 0.000 |
| 420 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т24: 317 → 0; т25: 0 → 317 | 0.000 |
| 421 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т7: 98 → 0; т8: 0 → 98 | 0.000 |
| 422 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т34: 0 → 94; т35: 94 → 0 | 0.035 |
| 423 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т47: 84 → 0 | 0.000 |
| 424 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т20: 433 → 0; т22: 0 → 433 | 0.000 |
| 425 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т26: 79 → 119 | 0.054 |
| 426 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т26: 119 → 178 | 0.027 |
| 427 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т26: 178 → 267 | 0.040 |
| 428 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т35: 78 → 117 | 0.051 |
| 429 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т48: 77 → 115 | 0.000 |
| 430 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 71 → 106 | 0.000 |
| 431 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т48: 63 → 0 | 0.000 |
| 432 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т22: 67 → 100 | 0.126 |
| 433 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т25: 317 → 383; т26: 66 → 0 | 0.002 |
| 434 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т36: 269 → 0; т37: 1 → 270 | 0.000 |
| 435 | shift-2 | valve | crude | `tg.tb.term_jp.grid_jp` | т8: 0 → 59; т10: 59 → 0 | 0.000 |
| 436 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т36: 52 → 0 | 0.000 |
| 437 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т1: 54 → 0; т3: 0 → 54 | 0.000 |
| 438 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т48: 47 → 0 | 0.000 |
| 439 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т23: 43 → 65 | 0.008 |
| 440 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т23: 65 → 97 | 0.012 |
| 441 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т23: 97 → 146 | 0.012 |
| 442 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 46 → 69 | 0.095 |
| 443 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 69 → 103 | 0.119 |
| 444 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т42: 45 → 67 | 0.002 |
| 445 | half-1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т41: 0 → 34; т42: 67 → 34 | 0.049 |
| 446 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т42: 34 → 50 | 0.047 |
| 447 | half | valve | crude | `tg.tb.term_tw.grid_tw` | т47: 45 → 22 | 0.016 |
| 448 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т47: 22 → 34 | 0.030 |
| 449 | half-1 | valve | crude | `tg.tb.term_tw.grid_tw` | т46: 19 → 36; т47: 34 → 17 | 0.004 |
| 450 | half | valve | crude | `tg.tb.term_kr.grid_kr` | т35: 44 → 22 | 0.000 |
| 451 | half | valve | crude | `tg.tb.term_kr.grid_kr` | т35: 22 → 11 | 0.000 |
| 452 | half | valve | crude | `tg.tb.term_kr.grid_kr` | т35: 11 → 5 | 0.000 |
| 453 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т35: 0 → 229; т37: 229 → 0 | 0.007 |
| 454 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т24: 36 → 54 | 0.022 |
| 455 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т24: 54 → 80 | 0.007 |
| 456 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т24: 80 → 0; т25: 137 → 217 | 0.010 |
| 457 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т47: 33 → 0 | 0.000 |
| 458 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т48: 33 → 0 | 0.000 |
| 459 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т49: 33 → 16 | 0.000 |
| 460 | half+1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т51: 33 → 16; т52: 0 → 16 | 0.000 |
| 461 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т17: 28 → 0; т18: 0 → 28 | 0.000 |
| 462 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т17: 28 → 0; т18: 110 → 138 | 0.001 |
| 463 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т21: 0 → 25; т22: 25 → 0 | 0.039 |
| 464 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 23 → 11; т46: 0 → 11 | 0.000 |
| 465 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 11 → 6; т46: 11 → 17 | 0.000 |
| 466 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 6 → 3; т46: 17 → 20 | 0.000 |
| 467 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т20: 13 → 0 | 0.025 |
| 468 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т5: 0 → 5; т6: 11 → 5 | 0.002 |
| 469 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т6: 5 → 8 | 0.015 |
| 470 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т25: 7 → 0 | 0.002 |
| 471 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т51: 0 → 0 | 0.000 |
| 472 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т51: 0 → 0 | 0.000 |
| 473 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т51: 0 → 0 | 0.000 |
| 474 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 0 → 0 | 0.000 |
| 475 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 0 → 0 | 0.000 |
| 476 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 0 → 0 | 0.000 |
| 477 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т25: 8,280 → 12,421 | 0.000 |
| 478 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т24: 5,491 → 11,702; т25: 12,421 → 6,210 | 0.000 |
| 479 | shift+1 | valve | lng | `tg.tb.term_eu.grid_eu` | т45: 7,204 → 0; т46: 5,036 → 12,240 | 0.003 |
| 480 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т39: 5,520 → 8,280 | 0.000 |
| 481 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т26: 5,491 → 8,237 | 0.000 |
| 482 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т37: 5,491 → 8,237 | 0.000 |
| 483 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т23: 4,953 → 10,804; т24: 11,702 → 5,851 | 0.000 |
| 484 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т31: 5,448 → 8,173 | 0.000 |
| 485 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т38: 5,408 → 8,112 | 0.000 |
| 486 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т3: 5,400 → 8,100 | 0.000 |
| 487 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т31: 5,400 → 2,700; т32: 0 → 2,700 | 0.000 |
| 488 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т36: 5,344 → 8,016 | 0.000 |
| 489 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т47: 5,071 → 7,607 | 0.022 |
| 490 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т45: 0 → 6,120; т46: 12,240 → 6,120 | 0.001 |
| 491 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т35: 5,029 → 7,544 | 0.000 |
| 492 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т44: 5,009 → 7,513 | 0.000 |
| 493 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т22: 4,953 → 10,355; т23: 10,804 → 5,402 | 0.000 |
| 494 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т21: 2,952 → 8,130; т22: 10,355 → 5,178 | 0.000 |
| 495 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т22: 5,178 → 7,767 | 0.000 |
| 496 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т33: 4,653 → 6,980 | 0.000 |
| 497 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т43: 4,653 → 6,980 | 0.000 |
| 498 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т34: 4,653 → 6,980 | 0.000 |
| 499 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т8: 59 → 4,616; т9: 4,556 → 0 | 0.000 |
| 500 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т8: 2,952 → 4,428 | 0.000 |
| 501 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т19: 2,952 → 4,428 | 0.000 |
| 502 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т20: 2,952 → 4,428 | 0.000 |
| 503 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т20: 4,428 → 6,642 | 0.000 |
| 504 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т7: 2,952 → 4,428 | 0.000 |
| 505 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т28: 2,916 → 0; т29: 0 → 2,916 | 0.033 |
| 506 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т18: 2,668 → 4,002 | 0.471 |
| 507 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т26: 2,668 → 4,002 | 0.213 |
| 508 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т35: 2,668 → 4,002 | 0.049 |
| 509 | x2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т36: 2,668 → 5,336 | 0.747 |
| 510 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т36: 5,336 → 8,005 | 1.225 |
| 511 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т45: 2,668 → 0; т46: 1,096 → 3,764 | 0.001 |
| 512 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т3: 2,439 → 1,219; т4: 0 → 1,219 | 0.000 |
| 513 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т3: 1,219 → 1,829 | 0.157 |
| 514 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т11: 2,416 → 3,624 | 0.151 |
| 515 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т6: 2,387 → 1,193; т7: 0 → 1,193 | 0.000 |
| 516 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т6: 1,193 → 597; т7: 1,193 → 1,790 | 0.000 |
| 517 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т6: 597 → 298 | 0.004 |
| 518 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т1: 2,251 → 1,126; т2: 0 → 1,126 | 0.816 |
| 519 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т26: 2,193 → 3,289 | 0.093 |
| 520 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т33: 2,036 → 0; т34: 0 → 2,036 | 0.000 |
| 521 | drop | order | lng | `sea.tb.src_us_lng.term_eu` | т51: 1,985 → 0 | 0.000 |
| 522 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т4: 1,900 → 0; т6: 104 → 2,004 | 0.001 |
| 523 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т15: 1,900 → 950 | 0.007 |
| 524 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 2,025 → 3,038 | 0.000 |
| 525 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т10: 0 → 1,519; т11: 3,038 → 1,519 | 0.000 |
| 526 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 1,519 → 759; т12: 162 → 921 | 0.000 |
| 527 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т49: 1,856 → 2,784 | 0.004 |
| 528 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т27: 536 → 1,443; т28: 1,814 → 907 | 0.000 |
| 529 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т12: 1,668 → 0; т13: 0 → 1,668 | 0.000 |
| 530 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т14: 1,501 → 0; т15: 188 → 1,688 | 0.000 |
| 531 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т45: 1,465 → 2,197 | 0.007 |
| 532 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 385 → 1,928; т46: 1,543 → 0 | 0.025 |
| 533 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т40: 1,382 → 2,072 | 0.835 |
| 534 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т3: 1,336 → 2,004 | 0.080 |
| 535 | half+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т3: 2,004 → 1,002; т4: 89 → 1,091 | 0.037 |
| 536 | half | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т3: 1,002 → 501 | 0.363 |
| 537 | half-1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т4: 1,091 → 1,759; т5: 1,336 → 668 | 0.073 |
| 538 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т10: 1,334 → 0; т12: 0 → 1,334 | 0.001 |
| 539 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т25: 1,264 → 1,896 | 0.026 |
| 540 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т13: 1,350 → 675; т14: 162 → 837 | 0.000 |
| 541 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т15: 1,350 → 675; т16: 0 → 675 | 0.000 |
| 542 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т15: 675 → 1,012 | 0.000 |
| 543 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т18: 1,350 → 675; т19: 364 → 1,039 | 0.000 |
| 544 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т18: 675 → 1,012 | 0.000 |
| 545 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т50: 1,254 → 0; т51: 0 → 1,254 | 0.421 |
| 546 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т44: 1,242 → 1,862 | 0.300 |
| 547 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т49: 1,242 → 0; т50: 0 → 1,242 | 0.251 |
| 548 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т36: 1,162 → 1,776; т37: 1,227 → 614 | 0.112 |
| 549 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т37: 614 → 920 | 0.124 |
| 550 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т43: 1,217 → 1,826 | 0.276 |
| 551 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т43: 1,826 → 2,739 | 0.449 |
| 552 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т43: 2,739 → 4,109 | 0.290 |
| 553 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т48: 1,217 → 0; т49: 0 → 1,217 | 0.196 |
| 554 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т14: 1,183 → 1,774 | 0.621 |
| 555 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т47: 1,169 → 1,754 | 0.094 |
| 556 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т47: 1,754 → 0; т48: 0 → 1,754 | 0.125 |
| 557 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т36: 1,776 → 888 | 0.019 |
| 558 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т36: 888 → 1,332 | 0.001 |
| 559 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 1,161 → 1,741 | 0.037 |
| 560 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 1,122 → 561; т22: 0 → 561 | 0.056 |
| 561 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 561 → 842 | 0.371 |
| 562 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т20: 0 → 421; т21: 842 → 421 | 0.073 |
| 563 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т46: 1,102 → 1,653 | 0.014 |
| 564 | x1.5 | order | lng | `sea.tb.src_us_lng.term_eu` | т46: 1,062 → 1,593 | 0.019 |
| 565 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т44: 1,053 → 527; т45: 0 → 527 | 0.004 |
| 566 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т21: 859 → 1,357; т22: 996 → 498 | 0.028 |
| 567 | shift-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т34: 0 → 1,039; т35: 1,039 → 0 | 0.000 |
| 568 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т15: 1,033 → 1,549 | 0.024 |
| 569 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 122 → 1,671; т15: 1,549 → 0 | 0.000 |
| 570 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т14: 1,774 → 2,721; т16: 946 → 0 | 0.104 |
| 571 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 957 → 478; т7: 456 → 934 | 0.000 |
| 572 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 478 → 239; т7: 934 → 1,173 | 0.031 |
| 573 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 239 → 120 | 0.001 |
| 574 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т8: 942 → 471; т9: 486 → 957 | 0.000 |
| 575 | shift-1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т5: 0 → 876; т6: 876 → 0 | 0.019 |
| 576 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т17: 300 → 734; т18: 867 → 434 | 0.112 |
| 577 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т19: 867 → 1,301; т20: 867 → 434 | 0.029 |
| 578 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т32: 867 → 434; т33: 1,301 → 1,734 | 0.075 |
| 579 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т32: 434 → 650 | 0.097 |
| 580 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т33: 1,734 → 2,168; т34: 867 → 434 | 0.023 |
| 581 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т35: 867 → 434; т36: 0 → 434 | 0.143 |
| 582 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т34: 434 → 650; т35: 434 → 217 | 0.250 |
| 583 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т10: 855 → 1,710; т11: 855 → 0 | 0.001 |
| 584 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т40: 841 → 1,262 | 0.000 |
| 585 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т40: 1,262 → 0; т41: 735 → 1,997 | 0.000 |
| 586 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т37: 900 → 0; т38: 0 → 900 | 0.000 |
| 587 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т45: 831 → 1,247 | 0.268 |
| 588 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т10: 805 → 0; т11: 3,624 → 4,429 | 0.000 |
| 589 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 859 → 429; т19: 0 → 429 | 0.000 |
| 590 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 429 → 644 | 0.122 |
| 591 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 644 → 966 | 0.006 |
| 592 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т44: 551 → 949; т45: 796 → 398 | 0.339 |
| 593 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т45: 398 → 597 | 0.121 |
| 594 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 850 → 425 | 0.003 |
| 595 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 425 → 0; т6: 120 → 545 | 0.000 |
| 596 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_jp]` | т3: 789 → 1,183 | 0.158 |
| 597 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т12: 785 → 392; т13: 785 → 1,177 | 0.000 |
| 598 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т10: 1,509 → 1,902; т12: 392 → 0 | 0.015 |
| 599 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т15: 785 → 392; т16: 785 → 1,177 | 0.000 |
| 600 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т15: 392 → 196; т16: 1,177 → 1,373 | 0.000 |
| 601 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т27: 785 → 1,177; т28: 785 → 392 | 0.062 |
| 602 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т43: 766 → 1,149 | 0.001 |
| 603 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т43: 1,149 → 1,724 | 0.001 |
| 604 | shift-1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т42: 798 → 2,522; т43: 1,724 → 0 | 0.011 |
| 605 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 818 → 1,226 | 0.000 |
| 606 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 1,226 → 613 | 0.000 |
| 607 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 804 → 1,206 | 0.000 |
| 608 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т40: 0 → 998; т41: 1,997 → 998 | 0.000 |
| 609 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т41: 998 → 0; т42: 735 → 1,733 | 0.000 |
| 610 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т41: 0 → 867; т42: 1,733 → 867 | 0.000 |
| 611 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т43: 735 → 1,102 | 0.000 |
| 612 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т43: 1,102 → 551; т44: 735 → 1,286 | 0.000 |
| 613 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т43: 551 → 827 | 0.000 |
| 614 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т44: 1,286 → 643; т45: 735 → 1,378 | 0.000 |
| 615 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т44: 643 → 964 | 0.000 |
| 616 | shift+2 | valve | lng | `tg.tb.term_jp.grid_jp` | т45: 1,378 → 0; т47: 0 → 1,378 | 0.002 |
| 617 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 712 → 1,069 | 0.027 |
| 618 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 1,069 → 1,603 | 0.036 |
| 619 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 1,603 → 2,405 | 0.047 |
| 620 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т13: 710 → 0; т15: 0 → 710 | 0.000 |
| 621 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т24: 346 → 1,103; т25: 756 → 0 | 0.024 |
| 622 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т38: 554 → 921; т39: 734 → 367 | 0.088 |
| 623 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 675 → 337; т19: 645 → 982 | 0.112 |
| 624 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т11: 0 → 349; т12: 698 → 349 | 0.047 |
| 625 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т12: 349 → 0; т14: 0 → 349 | 0.000 |
| 626 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 612 → 306; т35: 0 → 306 | 0.005 |
| 627 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 306 → 459 | 0.045 |
| 628 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т6: 0 → 573; т7: 573 → 0 | 0.006 |
| 629 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т43: 593 → 890 | 0.000 |
| 630 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т42: 534 → 800 | 0.264 |
| 631 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т3: 547 → 0; т4: 0 → 547 | 0.000 |
| 632 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 1,123 → 1,625; т10: 503 → 0 | 0.063 |
| 633 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 489 → 0; т45: 0 → 489 | 0.000 |
| 634 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т10: 488 → 0; т12: 0 → 488 | 0.000 |
| 635 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т27: 475 → 237 | 0.036 |
| 636 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т27: 237 → 356 | 0.251 |
| 637 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т32: 882 → 1,120; т33: 475 → 237 | 0.035 |
| 638 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т34: 475 → 712; т35: 475 → 237 | 0.003 |
| 639 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т35: 237 → 356 | 0.003 |
| 640 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т37: 475 → 950; т38: 475 → 0 | 0.029 |
| 641 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т39: 475 → 712; т40: 475 → 237 | 0.044 |
| 642 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т40: 237 → 0; т42: 475 → 712 | 0.004 |
| 643 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т40: 0 → 237; т41: 475 → 237 | 0.052 |
| 644 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т41: 237 → 0; т43: 475 → 712 | 0.000 |
| 645 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т41: 0 → 356; т42: 712 → 356 | 0.013 |
| 646 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т42: 356 → 178; т43: 712 → 891 | 0.005 |
| 647 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т42: 178 → 89; т43: 891 → 980 | 0.000 |
| 648 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т21: 486 → 729 | 0.002 |
| 649 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т36: 486 → 972; т37: 486 → 0 | 0.029 |
| 650 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т37: 0 → 243; т38: 486 → 243 | 0.053 |
| 651 | x2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т38: 243 → 486 | 0.016 |
| 652 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т39: 486 → 243 | 0.000 |
| 653 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т38: 486 → 608; т39: 243 → 122 | 0.024 |
| 654 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т46: 486 → 0 | 0.000 |
| 655 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т27: 448 → 224; т28: 412 → 636 | 0.072 |
| 656 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т27: 224 → 336 | 0.019 |
| 657 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т27: 336 → 168; т28: 636 → 804 | 0.088 |
| 658 | half | valve | crude | `tg.tb.term_kr.grid_kr` | т39: 474 → 237 | 0.006 |
| 659 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т19: 466 → 0; т20: 0 → 466 | 0.000 |
| 660 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т27: 466 → 698 | 0.130 |
| 661 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т27: 698 → 0; т28: 233 → 931 | 0.000 |
| 662 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т40: 431 → 646 | 0.668 |
| 663 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т28: 423 → 0; т29: 423 → 846 | 0.131 |
| 664 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т4: 450 → 0; т5: 0 → 450 | 0.000 |
| 665 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т2: 449 → 674 | 0.000 |
| 666 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т44: 324 → 732; т45: 407 → 0 | 0.011 |
| 667 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т22: 433 → 216; т23: 0 → 216 | 0.000 |
| 668 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т22: 216 → 108; т23: 216 → 324 | 0.000 |
| 669 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т22: 108 → 54; т23: 324 → 378 | 0.000 |
| 670 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т37: 400 → 200; т38: 400 → 600 | 0.351 |
| 671 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т37: 200 → 300 | 0.280 |
| 672 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т36: 390 → 585 | 0.169 |
| 673 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т43: 412 → 0; т44: 195 → 607 | 0.076 |
| 674 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т33: 399 → 0; т34: 0 → 399 | 0.000 |
| 675 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т4: 180 → 577; т5: 396 → 0 | 0.031 |
| 676 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т22: 360 → 180 | 0.076 |
| 677 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т22: 180 → 270 | 0.138 |
| 678 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т39: 367 → 560; т40: 386 → 193 | 0.128 |
| 679 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 1,928 → 2,891 | 0.000 |
| 680 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 2,891 → 4,337 | 0.000 |
| 681 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 4,337 → 6,506 | 0.000 |
| 682 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т41: 346 → 520 | 0.126 |
| 683 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т33: 369 → 0; т34: 0 → 369 | 0.000 |
| 684 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 346 → 173; т34: 0 → 173 | 0.000 |
| 685 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 173 → 0; т35: 243 → 416 | 0.000 |
| 686 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т10: 333 → 167; т11: 0 → 167 | 0.028 |
| 687 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т10: 300 → 450 | 0.076 |
| 688 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т12: 300 → 450 | 0.016 |
| 689 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т17: 734 → 367 | 0.371 |
| 690 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т17: 367 → 550 | 0.278 |
| 691 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т16: 300 → 575; т17: 550 → 275 | 0.107 |
| 692 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т26: 267 → 134 | 0.104 |
| 693 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т36: 0 → 270; т37: 270 → 0 | 0.021 |
| 694 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т10: 262 → 0; т12: 0 → 262 | 0.000 |
| 695 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т16: 0 → 237; т17: 237 → 0 | 0.004 |
| 696 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т25: 217 → 0; т27: 168 → 386 | 0.164 |
| 697 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т7: 233 → 116; т8: 233 → 349 | 0.005 |
| 698 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т28: 931 → 0; т30: 233 → 1,164 | 0.000 |
| 699 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т29: 233 → 349 | 0.008 |
| 700 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т34: 233 → 116; т35: 229 → 346 | 0.000 |
| 701 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т11: 167 → 282; т12: 230 → 115 | 0.001 |
| 702 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т12: 115 → 0; т14: 0 → 115 | 0.000 |
| 703 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т18: 204 → 0; т19: 264 → 468 | 0.050 |
| 704 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т25: 0 → 214; т27: 214 → 0 | 0.001 |
| 705 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т39: 0 → 91; т40: 183 → 91 | 0.005 |
| 706 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т4: 577 → 288; т5: 0 → 288 | 0.000 |
| 707 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т27: 170 → 85; т28: 0 → 85 | 0.000 |
| 708 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 581 → 737; т25: 157 → 0 | 0.013 |
| 709 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т23: 146 → 73 | 0.013 |
| 710 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т9: 148 → 0; т10: 0 → 148 | 0.000 |
| 711 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т8: 98 → 172; т9: 148 → 74 | 0.000 |
| 712 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т9: 74 → 37; т10: 768 → 805 | 0.001 |
| 713 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т8: 172 → 191; т9: 37 → 18 | 0.001 |
| 714 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т22: 0 → 134; т24: 134 → 0 | 0.004 |
| 715 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 1,671 → 0; т15: 0 → 1,671 | 0.000 |
| 716 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т48: 115 → 173 | 0.000 |
| 717 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т18: 114 → 171 | 0.027 |
| 718 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т6: 2,004 → 1,002; т7: 0 → 1,002 | 0.000 |
| 719 | shift+2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т22: 109 → 0; т24: 0 → 109 | 0.000 |
| 720 | shift+2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т23: 109 → 0; т25: 0 → 109 | 0.000 |
| 721 | shift+2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т26: 109 → 0; т28: 0 → 109 | 0.000 |
| 722 | shift+2 | valve | crude | `tg.tb.term_tw.grid_tw` | т45: 109 → 0; т47: 17 → 126 | 0.049 |
| 723 | shift+1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т36: 106 → 0; т37: 0 → 106 | 0.000 |
| 724 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 106 → 159 | 0.000 |
| 725 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 159 → 238 | 0.000 |
| 726 | shift-2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т43: 77 → 178; т45: 101 → 0 | 0.012 |
| 727 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т34: 101 → 151 | 0.015 |
| 728 | x1.5 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т2: 89 → 134 | 0.026 |
| 729 | shift+1 | order | lng | `sea.tb.src_us_lng.chk_panama [lane.src_us_lng.term_tw]` | т2: 134 → 0; т3: 501 → 635 | 0.005 |
| 730 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т34: 94 → 0; т35: 0 → 94 | 0.000 |
| 731 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т44: 93 → 139 | 0.000 |
| 732 | half+1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т44: 139 → 70; т45: 0 → 70 | 0.000 |
| 733 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т44: 70 → 104 | 0.000 |
| 734 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т9: 87 → 0; т10: 0 → 87 | 0.000 |
| 735 | half-1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т42: 50 → 139; т43: 178 → 89 | 0.038 |
| 736 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т7: 0 → 2,308; т8: 4,616 → 2,308 | 0.000 |
| 737 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т8: 2,308 → 1,154; т9: 0 → 1,154 | 0.000 |
| 738 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т3: 54 → 0; т4: 0 → 54 | 0.000 |
| 739 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 49 → 24; т43: 0 → 24 | 0.000 |
| 740 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 24 → 12; т43: 24 → 37 | 0.000 |
| 741 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т41: 0 → 6; т42: 12 → 6 | 0.000 |
| 742 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т42: 139 → 70 | 0.000 |
| 743 | x1.5 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т42: 70 → 104 | 0.003 |
| 744 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т17: 533 → 555; т18: 45 → 22 | 0.000 |
| 745 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т41: 34 → 0 | 0.001 |
| 746 | shift-1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т45: 70 → 102; т46: 33 → 0 | 0.000 |
| 747 | shift-2 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т49: 16 → 33; т51: 16 → 0 | 0.000 |
| 748 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т6: 8 → 0; т7: 116 → 125 | 0.000 |
| 749 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т5: 5 → 0; т6: 0 → 5 | 0.000 |
| 750 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 3 → 4 | 0.000 |
| 751 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 4 → 6 | 0.000 |
| 752 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 6 → 10 | 0.000 |
| 753 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 0 → 0; т48: 0 → 0 | 0.000 |
| 754 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 6,506 → 3,253 | 0.000 |
| 755 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т41: 5,372 → 8,058 | 0.000 |
| 756 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т48: 3,952 → 5,928 | 0.004 |
| 757 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т23: 1,925 → 4,675; т24: 2,750 → 0 | 0.000 |
| 758 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т32: 2,700 → 1,350; т33: 0 → 1,350 | 0.000 |
| 759 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т32: 1,350 → 2,025 | 0.000 |

## Епізод 2: J 2193.09 → 2149.28 млрд USD (−43.81), прийнято 610 ходів

| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |
|---|---|---|---|---|---|---|
| 1 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т42: 11,534 → 17,301 | 0.000 |
| 2 | half+1 | valve | lng | `tg.tb.term_eu.grid_eu` | т42: 17,301 → 8,651; т43: 5,898 → 14,549 | 0.000 |
| 3 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т21: 8,100 → 12,150 | 0.000 |
| 4 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 0 → 12,150; т21: 12,150 → 0 | 1.557 |
| 5 | drop | order | lng | `sea.tb.src_us_lng.term_eu` | т1: 7,940 → 0 | 0.013 |
| 6 | half+1 | order | lng | `sea.tb.src_us_lng.term_eu` | т47: 7,940 → 3,970; т48: 3,446 → 7,416 | 0.000 |
| 7 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 0 → 6,480; т35: 6,480 → 0 | 0.749 |
| 8 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т1: 1,069 → 6,469; т2: 5,400 → 0 | 1.792 |
| 9 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т3: 5,400 → 0; т4: 5,364 → 10,764 | 0.199 |
| 10 | shift+2 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 5,400 → 0; т25: 0 → 5,400 | 0.000 |
| 11 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 0 → 2,700; т36: 5,400 → 2,700 | 0.000 |
| 12 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 2,700 → 4,050; т36: 2,700 → 1,350 | 0.000 |
| 13 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т4: 10,764 → 16,146 | 0.001 |
| 14 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т10: 3,182 → 4,773 | 0.000 |
| 15 | shift+2 | valve | lng | `tg.tb.term_kr.grid_kr` | т10: 4,773 → 0; т12: 2,750 → 7,523 | 0.606 |
| 16 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т21: 3,182 → 0; т22: 2,750 → 5,932 | 0.656 |
| 17 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т33: 3,182 → 1,591; т34: 1,925 → 3,516 | 0.340 |
| 18 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т32: 0 → 1,591; т33: 1,591 → 0 | 0.000 |
| 19 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т2: 2,829 → 4,243 | 0.000 |
| 20 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т35: 2,750 → 1,375; т36: 2,534 → 3,909 | 0.955 |
| 21 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т10: 2,706 → 4,059 | 0.000 |
| 22 | shift-2 | valve | lng | `tg.tb.term_tw.grid_tw` | т8: 0 → 4,059; т10: 4,059 → 0 | 0.105 |
| 23 | shift-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т18: 0 → 2,706; т19: 2,706 → 0 | 0.649 |
| 24 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т4: 2,594 → 3,891 | 0.000 |
| 25 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т36: 3,909 → 1,954; т37: 862 → 2,816 | 0.456 |
| 26 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т24: 2,245 → 3,368 | 0.000 |
| 27 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т24: 3,368 → 5,051 | 0.000 |
| 28 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 0 → 5,051; т24: 5,051 → 0 | 0.000 |
| 29 | drop | valve | lng | `tg.tb.term_tw.grid_tw` | т4: 2,177 → 0 | 0.425 |
| 30 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т13: 2,177 → 3,265; т14: 2,177 → 1,088 | 0.152 |
| 31 | shift+2 | valve | lng | `tg.tb.term_tw.grid_tw` | т14: 1,088 → 0; т16: 0 → 1,088 | 0.139 |
| 32 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т28: 2,177 → 3,265 | 0.517 |
| 33 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т29: 2,177 → 3,265; т30: 2,177 → 1,088 | 0.001 |
| 34 | half | valve | lng | `tg.tb.term_tw.grid_tw` | т30: 1,088 → 544 | 0.000 |
| 35 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т27: 2,128 → 3,192 | 0.000 |
| 36 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т18: 1,985 → 0; т19: 0 → 1,985 | 0.022 |
| 37 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т26: 1,900 → 0 | 0.002 |
| 38 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т29: 1,900 → 950 | 0.001 |
| 39 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т30: 1,900 → 950 | 0.001 |
| 40 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т42: 1,900 → 950; т43: 1,026 → 1,976 | 0.000 |
| 41 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т42: 950 → 0; т44: 0 → 950 | 0.000 |
| 42 | x2 | valve | lng | `tg.tb.term_tw.grid_tw` | т15: 1,676 → 3,351 | 0.095 |
| 43 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т14: 0 → 1,676; т15: 3,351 → 1,676 | 0.253 |
| 44 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т15: 1,676 → 838; т16: 1,088 → 1,926 | 0.000 |
| 45 | shift+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т3: 1,524 → 0; т4: 0 → 1,524 | 0.179 |
| 46 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т19: 0 → 762; т20: 1,524 → 762 | 0.000 |
| 47 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т20: 762 → 1,143 | 0.052 |
| 48 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т20: 1,143 → 1,714 | 0.411 |
| 49 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 1,444 → 2,165 | 0.001 |
| 50 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 2,165 → 3,248 | 0.001 |
| 51 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т10: 221 → 3,469; т11: 3,248 → 0 | 0.490 |
| 52 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 1,271 → 1,906 | 0.127 |
| 53 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 1,906 → 953; т6: 0 → 953 | 0.040 |
| 54 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 953 → 1,430 | 0.000 |
| 55 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т1: 1,262 → 1,893 | 2.579 |
| 56 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т1: 1,893 → 2,840 | 2.267 |
| 57 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т1: 2,840 → 4,259 | 0.029 |
| 58 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т15: 1,350 → 2,025 | 0.001 |
| 59 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т15: 2,025 → 3,038 | 0.000 |
| 60 | drop | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 1,350 → 0 | 0.767 |
| 61 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 1,350 → 2,025 | 0.000 |
| 62 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т26: 2,025 → 3,038 | 0.000 |
| 63 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т25: 175 → 1,694; т26: 3,038 → 1,519 | 0.000 |
| 64 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 1,350 → 2,025 | 0.000 |
| 65 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 359 → 1,371; т29: 2,025 → 1,012 | 0.000 |
| 66 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 1,012 → 506; т30: 162 → 668 | 0.000 |
| 67 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 1,350 → 675; т32: 162 → 837 | 0.000 |
| 68 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 675 → 1,012 | 0.000 |
| 69 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т30: 668 → 1,174; т31: 1,012 → 506 | 0.000 |
| 70 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т33: 1,350 → 675; т34: 162 → 837 | 0.000 |
| 71 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т33: 675 → 1,012 | 0.568 |
| 72 | drop | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 1,350 → 0 | 0.147 |
| 73 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 1,349 → 2,024 | 0.000 |
| 74 | half | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 2,024 → 1,012 | 1.020 |
| 75 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т6: 840 → 2,082; т7: 1,243 → 0 | 0.013 |
| 76 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т5: 1,215 → 0; т6: 0 → 1,215 | 1.267 |
| 77 | half | order | crude | `sea.tb.src_us_crude.term_eu` | т3: 1,248 → 624 | 0.001 |
| 78 | half-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т3: 624 → 1,248; т4: 1,248 → 624 | 0.001 |
| 79 | half-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т4: 624 → 1,248; т5: 1,248 → 624 | 0.001 |
| 80 | half | order | crude | `sea.tb.src_us_crude.term_eu` | т5: 624 → 312 | 0.000 |
| 81 | shift-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т5: 312 → 1,560; т6: 1,248 → 0 | 0.001 |
| 82 | shift-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т6: 0 → 1,248; т7: 1,248 → 0 | 0.001 |
| 83 | shift-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т7: 0 → 1,248; т8: 1,248 → 0 | 0.001 |
| 84 | shift-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т8: 0 → 1,248; т9: 1,248 → 0 | 0.000 |
| 85 | half | order | crude | `sea.tb.src_us_crude.term_eu` | т10: 1,248 → 624 | 0.002 |
| 86 | half-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т10: 624 → 1,248; т11: 1,248 → 624 | 0.001 |
| 87 | half | order | crude | `sea.tb.src_us_crude.term_eu` | т11: 624 → 312 | 0.000 |
| 88 | shift-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т11: 312 → 1,560; т12: 1,248 → 0 | 0.001 |
| 89 | half+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т42: 1,248 → 624; т43: 523 → 1,147 | 0.000 |
| 90 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т42: 624 → 0; т44: 204 → 828 | 0.000 |
| 91 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 1,038 → 1,557; т19: 1,038 → 519 | 0.012 |
| 92 | drop | order | lng | `sea.tb.src_ru_gas.term_jp` | т50: 1,028 → 0 | 0.000 |
| 93 | drop | order | lng | `sea.tb.src_ru_gas.term_jp` | т51: 1,028 → 0 | 0.000 |
| 94 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т43: 1,976 → 988; т44: 950 → 1,938 | 0.000 |
| 95 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т11: 1,038 → 519 | 0.003 |
| 96 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т1: 929 → 464 | 0.176 |
| 97 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т1: 464 → 697 | 0.059 |
| 98 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т51: 755 → 1,191; т52: 872 → 436 | 0.108 |
| 99 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т38: 871 → 436 | 0.192 |
| 100 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т38: 436 → 868; т39: 865 → 432 | 0.000 |
| 101 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т37: 2,816 → 4,224 | 0.068 |
| 102 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 855 → 1,282 | 0.272 |
| 103 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 1,282 → 1,924 | 0.359 |
| 104 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т1: 1,924 → 0; т2: 855 → 2,779 | 0.641 |
| 105 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т2: 2,779 → 1,389 | 0.272 |
| 106 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т2: 1,389 → 2,084 | 0.874 |
| 107 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т25: 855 → 427 | 0.002 |
| 108 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т42: 855 → 427; т43: 0 → 427 | 0.000 |
| 109 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т39: 432 → 858; т40: 851 → 425 | 0.000 |
| 110 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т6: 2,082 → 3,124 | 1.069 |
| 111 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 900 → 1,350 | 0.000 |
| 112 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 1,350 → 2,025 | 0.000 |
| 113 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т23: 450 → 1,350; т24: 900 → 0 | 0.000 |
| 114 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 0 → 450; т36: 900 → 450 | 0.000 |
| 115 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 450 → 675 | 0.000 |
| 116 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 675 → 1,012 | 0.000 |
| 117 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т5: 892 → 1,337 | 0.334 |
| 118 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т5: 1,337 → 2,006 | 0.203 |
| 119 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т5: 2,006 → 3,009 | 0.161 |
| 120 | shift+2 | valve | lng | `tg.tb.term_kr.grid_kr` | т47: 827 → 0; т49: 728 → 1,555 | 0.015 |
| 121 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т46: 816 → 0; т47: 0 → 816 | 0.000 |
| 122 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т1: 731 → 1,123; т2: 785 → 392 | 0.113 |
| 123 | x2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т2: 392 → 785 | 0.100 |
| 124 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т22: 295 → 1,080; т23: 785 → 0 | 0.000 |
| 125 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 785 → 1,177 | 0.023 |
| 126 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т25: 785 → 392; т26: 785 → 1,177 | 0.000 |
| 127 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т27: 785 → 392; т28: 785 → 1,177 | 0.000 |
| 128 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т29: 785 → 392; т30: 746 → 1,139 | 0.000 |
| 129 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т47: 785 → 0 | 0.001 |
| 130 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т48: 785 → 0 | 0.001 |
| 131 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т49: 785 → 0 | 0.001 |
| 132 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 781 → 1,171 | 0.163 |
| 133 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т44: 703 → 1,289; т45: 1,171 → 586 | 0.018 |
| 134 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 586 → 293 | 0.112 |
| 135 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т47: 816 → 1,595; т48: 779 → 0 | 0.002 |
| 136 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т46: 776 → 0 | 0.001 |
| 137 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т32: 775 → 388; т33: 769 → 1,156 | 0.000 |
| 138 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т31: 766 → 383; т32: 388 → 770 | 0.000 |
| 139 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т51: 1,191 → 1,787 | 0.089 |
| 140 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т34: 755 → 1,132 | 0.036 |
| 141 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т34: 1,132 → 1,698 | 0.078 |
| 142 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т34: 1,698 → 849; т35: 648 → 1,497 | 0.000 |
| 143 | drop | valve | lng | `tg.tb.term_tw.grid_tw` | т1: 752 → 0 | 0.474 |
| 144 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т30: 1,139 → 1,708 | 0.015 |
| 145 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т30: 1,708 → 2,562 | 0.102 |
| 146 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т29: 392 → 1,674; т30: 2,562 → 1,281 | 0.006 |
| 147 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т40: 425 → 797; т41: 744 → 372 | 0.000 |
| 148 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т41: 372 → 186 | 0.116 |
| 149 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т40: 720 → 0; т41: 731 → 1,451 | 0.000 |
| 150 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т12: 706 → 1,059 | 0.155 |
| 151 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т11: 440 → 970; т12: 1,059 → 529 | 0.013 |
| 152 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т12: 529 → 794 | 0.083 |
| 153 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т13: 706 → 0; т14: 706 → 1,412 | 0.000 |
| 154 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т14: 1,412 → 706; т15: 706 → 1,412 | 0.000 |
| 155 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т15: 1,412 → 706; т16: 706 → 1,412 | 0.000 |
| 156 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т16: 1,412 → 706; т17: 706 → 1,412 | 0.000 |
| 157 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т17: 1,412 → 706; т18: 534 → 1,240 | 0.000 |
| 158 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 664 → 1,308; т44: 1,289 → 644 | 0.000 |
| 159 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 1,308 → 1,631; т44: 644 → 322 | 0.000 |
| 160 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т49: 1,555 → 2,250; т50: 695 → 0 | 0.050 |
| 161 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т41: 186 → 877; т42: 691 → 0 | 0.000 |
| 162 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т39: 685 → 1,027 | 0.057 |
| 163 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т39: 1,027 → 0; т40: 0 → 1,027 | 0.000 |
| 164 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т42: 683 → 0 | 0.002 |
| 165 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т1: 667 → 1,001 | 0.000 |
| 166 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 667 → 334 | 0.012 |
| 167 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 334 → 167 | 0.000 |
| 168 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 167 → 83 | 0.000 |
| 169 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т21: 667 → 0; т23: 0 → 667 | 0.000 |
| 170 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т47: 667 → 0 | 0.000 |
| 171 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 1,631 → 2,446 | 0.023 |
| 172 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т42: 0 → 1,223; т43: 2,446 → 1,223 | 0.000 |
| 173 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 1,223 → 0; т44: 322 → 1,545 | 0.000 |
| 174 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т45: 659 → 0 | 0.001 |
| 175 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т10: 655 → 982 | 0.100 |
| 176 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т10: 982 → 1,474 | 0.130 |
| 177 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т9: 604 → 2,077; т10: 1,474 → 0 | 0.052 |
| 178 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 698 → 0 | 0.003 |
| 179 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т35: 1,497 → 0; т36: 595 → 2,092 | 0.159 |
| 180 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т5: 633 → 949 | 0.196 |
| 181 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т43: 632 → 0 | 0.001 |
| 182 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 613 → 919 | 0.046 |
| 183 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т38: 607 → 0; т39: 0 → 607 | 0.000 |
| 184 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т8: 596 → 1,635; т9: 2,077 → 1,039 | 0.079 |
| 185 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т9: 1,039 → 1,558 | 0.099 |
| 186 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т44: 599 → 0 | 0.001 |
| 187 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т20: 596 → 894 | 0.000 |
| 188 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т19: 596 → 0 | 0.000 |
| 189 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т36: 2,092 → 1,046; т37: 568 → 1,614 | 0.129 |
| 190 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т36: 1,046 → 0; т38: 0 → 1,046 | 0.000 |
| 191 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т52: 612 → 918 | 0.193 |
| 192 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 586 → 880 | 0.000 |
| 193 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 880 → 1,320 | 0.000 |
| 194 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т8: 0 → 1,320; т9: 1,320 → 0 | 0.000 |
| 195 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 546 → 273 | 0.001 |
| 196 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 273 → 136 | 0.000 |
| 197 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 136 → 68 | 0.000 |
| 198 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т2: 537 → 806 | 0.076 |
| 199 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т2: 806 → 0; т4: 0 → 806 | 0.076 |
| 200 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т6: 530 → 0; т7: 205 → 735 | 0.000 |
| 201 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т11: 40 → 563; т12: 523 → 0 | 0.017 |
| 202 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т1: 475 → 0 | 0.289 |
| 203 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т2: 475 → 712 | 0.627 |
| 204 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т2: 712 → 0; т4: 0 → 712 | 0.003 |
| 205 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 475 → 0; т7: 475 → 950 | 0.000 |
| 206 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 0 → 475; т6: 475 → 0 | 0.057 |
| 207 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т8: 475 → 237 | 0.001 |
| 208 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т8: 237 → 0; т9: 475 → 712 | 0.000 |
| 209 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т9: 712 → 356; т10: 475 → 831 | 0.000 |
| 210 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т10: 831 → 416; т11: 475 → 891 | 0.000 |
| 211 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т11: 891 → 445; т12: 475 → 920 | 0.000 |
| 212 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т12: 920 → 460; т13: 475 → 935 | 0.000 |
| 213 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т13: 935 → 468; т14: 475 → 943 | 0.000 |
| 214 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т14: 943 → 471; т15: 475 → 946 | 0.000 |
| 215 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т15: 946 → 473; т16: 475 → 948 | 0.000 |
| 216 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т16: 948 → 474; т17: 475 → 949 | 0.000 |
| 217 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т18: 475 → 0; т19: 475 → 950 | 0.000 |
| 218 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т19: 950 → 475; т20: 475 → 950 | 0.000 |
| 219 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т22: 490 → 0; т23: 0 → 490 | 0.000 |
| 220 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т21: 486 → 243; т22: 0 → 243 | 0.000 |
| 221 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т21: 243 → 122 | 0.044 |
| 222 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 486 → 729 | 0.857 |
| 223 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 729 → 1,094 | 1.147 |
| 224 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 1,094 → 1,640 | 0.078 |
| 225 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т2: 486 → 729 | 1.001 |
| 226 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т2: 729 → 1,094 | 1.294 |
| 227 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т2: 1,094 → 1,640 | 0.013 |
| 228 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т7: 486 → 729 | 0.113 |
| 229 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т7: 729 → 1,094 | 0.001 |
| 230 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 207 → 1,300; т7: 1,094 → 0 | 0.258 |
| 231 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т8: 486 → 729 | 1.001 |
| 232 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т8: 729 → 1,094 | 1.180 |
| 233 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т8: 1,094 → 1,640 | 0.108 |
| 234 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т9: 486 → 729 | 0.451 |
| 235 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т9: 729 → 1,094 | 0.329 |
| 236 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т9: 1,094 → 547; т10: 486 → 1,033 | 0.055 |
| 237 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т12: 486 → 0; т13: 486 → 972 | 0.000 |
| 238 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 972 → 486; т14: 486 → 972 | 0.000 |
| 239 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т14: 972 → 486; т15: 486 → 972 | 0.000 |
| 240 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т15: 972 → 486; т16: 486 → 972 | 0.000 |
| 241 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т16: 972 → 486; т17: 486 → 972 | 0.000 |
| 242 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 486 → 243; т19: 486 → 729 | 0.000 |
| 243 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 243 → 122 | 0.001 |
| 244 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 122 → 0; т19: 729 → 850 | 0.000 |
| 245 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т20: 486 → 0; т21: 0 → 486 | 0.000 |
| 246 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т47: 486 → 0 | 0.002 |
| 247 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т11: 970 → 0 | 0.008 |
| 248 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т23: 436 → 0 | 0.003 |
| 249 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т24: 436 → 0 | 0.003 |
| 250 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т25: 436 → 0 | 0.003 |
| 251 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 436 → 0 | 0.003 |
| 252 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т27: 436 → 0 | 0.003 |
| 253 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т28: 436 → 0 | 0.003 |
| 254 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т29: 436 → 0 | 0.003 |
| 255 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т30: 436 → 0 | 0.002 |
| 256 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т31: 436 → 0 | 0.002 |
| 257 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т32: 436 → 0 | 0.002 |
| 258 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 436 → 0 | 0.002 |
| 259 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т1: 428 → 641 | 0.006 |
| 260 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т1: 641 → 0; т2: 0 → 641 | 0.021 |
| 261 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т3: 423 → 0; т4: 423 → 847 | 0.000 |
| 262 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т23: 1,350 → 2,025 | 0.716 |
| 263 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т6: 0 → 411; т7: 411 → 0 | 0.007 |
| 264 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т6: 431 → 0; т7: 28 → 459 | 0.434 |
| 265 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 379 → 569 | 0.065 |
| 266 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 569 → 284 | 0.032 |
| 267 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т4: 94 → 378; т5: 284 → 0 | 0.027 |
| 268 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т21: 340 → 0 | 0.003 |
| 269 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т10: 0 → 355; т11: 355 → 0 | 0.348 |
| 270 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т28: 355 → 533 | 0.000 |
| 271 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т28: 533 → 799 | 0.000 |
| 272 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т34: 330 → 0 | 0.015 |
| 273 | half+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т1: 329 → 165; т2: 219 → 384 | 0.187 |
| 274 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 302 → 0 | 0.002 |
| 275 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т1: 300 → 450 | 0.097 |
| 276 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т2: 300 → 0 | 0.001 |
| 277 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т45: 300 → 0 | 0.000 |
| 278 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т46: 300 → 0 | 0.000 |
| 279 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т47: 300 → 0 | 0.000 |
| 280 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 299 → 149 | 0.004 |
| 281 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т22: 1,080 → 540; т23: 0 → 540 | 0.000 |
| 282 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т11: 745 → 875; т12: 262 → 131 | 0.089 |
| 283 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т12: 131 → 0; т13: 785 → 916 | 0.000 |
| 284 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т22: 270 → 135; т23: 270 → 405 | 0.001 |
| 285 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т24: 270 → 135; т25: 270 → 405 | 0.000 |
| 286 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т26: 270 → 135; т27: 270 → 405 | 0.000 |
| 287 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т28: 270 → 135; т29: 270 → 405 | 0.000 |
| 288 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т30: 270 → 135; т31: 270 → 405 | 0.000 |
| 289 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т32: 270 → 135; т33: 270 → 405 | 0.000 |
| 290 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т34: 270 → 135; т35: 270 → 405 | 0.000 |
| 291 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 270 → 135; т37: 270 → 405 | 0.000 |
| 292 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т38: 270 → 135; т39: 270 → 405 | 0.000 |
| 293 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т40: 270 → 135; т41: 270 → 405 | 0.000 |
| 294 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 270 → 135; т43: 270 → 405 | 0.000 |
| 295 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т41: 405 → 810; т43: 405 → 0 | 0.000 |
| 296 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т44: 270 → 135; т45: 270 → 405 | 0.000 |
| 297 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т46: 270 → 135; т47: 126 → 261 | 0.000 |
| 298 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 233 → 349 | 0.331 |
| 299 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 349 → 524 | 0.170 |
| 300 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т17: 233 → 495; т18: 524 → 262 | 0.008 |
| 301 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т19: 233 → 466; т20: 233 → 0 | 0.001 |
| 302 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т12: 233 → 349 | 0.017 |
| 303 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т12: 349 → 0; т13: 233 → 582 | 0.000 |
| 304 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т13: 582 → 0; т14: 233 → 815 | 0.000 |
| 305 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т14: 815 → 407; т15: 233 → 640 | 0.000 |
| 306 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т14: 407 → 204; т15: 640 → 844 | 0.000 |
| 307 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т8: 233 → 0 | 0.217 |
| 308 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т10: 233 → 349 | 0.030 |
| 309 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т43: 230 → 345 | 0.000 |
| 310 | half+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т43: 345 → 172; т44: 191 → 363 | 0.000 |
| 311 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т43: 172 → 258 | 0.000 |
| 312 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т1: 224 → 336 | 0.001 |
| 313 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т1: 336 → 504 | 0.001 |
| 314 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 0 → 1,735; т10: 3,469 → 1,735 | 0.202 |
| 315 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т10: 1,735 → 0; т11: 0 → 1,735 | 0.000 |
| 316 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т12: 195 → 415; т13: 220 → 0 | 0.118 |
| 317 | shift-2 | valve | crude | `tg.tb.term_kr.grid_kr` | т12: 415 → 635; т14: 220 → 0 | 0.001 |
| 318 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т16: 220 → 0 | 0.028 |
| 319 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т37: 217 → 0; т38: 178 → 394 | 0.000 |
| 320 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т45: 212 → 0; т46: 0 → 212 | 0.000 |
| 321 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т23: 208 → 0; т24: 178 → 386 | 0.003 |
| 322 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т44: 828 → 0; т45: 0 → 828 | 0.000 |
| 323 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т41: 150 → 251; т42: 203 → 102 | 0.000 |
| 324 | shift-2 | valve | crude | `tg.tb.term_kr.grid_kr` | т40: 43 → 145; т42: 102 → 0 | 0.000 |
| 325 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т50: 199 → 299 | 0.000 |
| 326 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т49: 165 → 464; т50: 299 → 0 | 0.000 |
| 327 | shift+2 | valve | crude | `tg.tb.term_kr.grid_kr` | т44: 363 → 0; т46: 77 → 440 | 0.000 |
| 328 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т36: 190 → 0; т37: 0 → 190 | 0.000 |
| 329 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т44: 186 → 0 | 0.001 |
| 330 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т14: 184 → 0; т16: 88 → 272 | 0.558 |
| 331 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т24: 386 → 0; т25: 116 → 502 | 0.006 |
| 332 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 394 → 197; т39: 100 → 297 | 0.000 |
| 333 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 197 → 99; т39: 297 → 396 | 0.000 |
| 334 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 99 → 49; т39: 396 → 445 | 0.000 |
| 335 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т18: 172 → 0 | 0.002 |
| 336 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т18: 172 → 0; т20: 73 → 245 | 0.000 |
| 337 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т2: 169 → 0 | 0.000 |
| 338 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т48: 115 → 347; т49: 464 → 232 | 0.000 |
| 339 | half+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т49: 232 → 116; т50: 0 → 116 | 0.000 |
| 340 | shift-2 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 67 → 183; т49: 116 → 0 | 0.000 |
| 341 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т11: 163 → 245 | 0.000 |
| 342 | shift-1 | valve | crude | `tg.tb.term_tw.grid_tw` | т10: 33 → 278; т11: 245 → 0 | 0.114 |
| 343 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т1: 163 → 244 | 0.120 |
| 344 | shift+1 | valve | crude | `tg.tb.term_tw.grid_tw` | т1: 244 → 0; т2: 108 → 352 | 0.092 |
| 345 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т21: 160 → 0 | 0.011 |
| 346 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т43: 152 → 0 | 0.001 |
| 347 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т41: 251 → 0; т42: 0 → 251 | 0.000 |
| 348 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т51: 138 → 0; т52: 22 → 160 | 0.408 |
| 349 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т35: 137 → 0; т36: 0 → 137 | 0.000 |
| 350 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т22: 134 → 0 | 0.001 |
| 351 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 125 → 0 | 0.001 |
| 352 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 109 → 0 | 0.000 |
| 353 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т35: 109 → 0 | 0.000 |
| 354 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т36: 109 → 0 | 0.000 |
| 355 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 109 → 0 | 0.000 |
| 356 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т38: 109 → 0 | 0.000 |
| 357 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т39: 109 → 0 | 0.000 |
| 358 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т40: 109 → 0 | 0.000 |
| 359 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т41: 109 → 0 | 0.000 |
| 360 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 109 → 0 | 0.000 |
| 361 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 109 → 0 | 0.000 |
| 362 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т44: 109 → 0 | 0.000 |
| 363 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т45: 109 → 0 | 0.000 |
| 364 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т46: 109 → 0 | 0.000 |
| 365 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т47: 109 → 0 | 0.000 |
| 366 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т48: 109 → 0 | 0.000 |
| 367 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т49: 109 → 0 | 0.000 |
| 368 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т45: 113 → 169 | 0.000 |
| 369 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т45: 169 → 254 | 0.000 |
| 370 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т44: 0 → 254; т45: 254 → 0 | 0.000 |
| 371 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т19: 110 → 0 | 0.017 |
| 372 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т20: 110 → 0 | 0.047 |
| 373 | half | valve | crude | `tg.tb.term_tw.grid_tw` | т3: 109 → 54 | 0.001 |
| 374 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т42: 102 → 153 | 0.035 |
| 375 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т42: 153 → 0; т43: 0 → 153 | 0.000 |
| 376 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т3: 94 → 47; т4: 378 → 425 | 0.000 |
| 377 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т16: 272 → 408 | 0.000 |
| 378 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т19: 88 → 0; т21: 88 → 175 | 0.000 |
| 379 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 175 → 263 | 0.000 |
| 380 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т45: 0 → 440; т46: 440 → 0 | 0.000 |
| 381 | shift+2 | valve | crude | `tg.tb.term_tw.grid_tw` | т14: 76 → 0; т16: 0 → 76 | 0.048 |
| 382 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т20: 245 → 0; т21: 122 → 367 | 0.000 |
| 383 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т5: 73 → 110 | 0.005 |
| 384 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т5: 110 → 165 | 0.007 |
| 385 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 183 → 274 | 0.000 |
| 386 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т46: 0 → 137; т47: 274 → 137 | 0.000 |
| 387 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т47: 137 → 206 | 0.000 |
| 388 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т39: 445 → 477; т40: 64 → 32 | 0.005 |
| 389 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т39: 477 → 494; т40: 32 → 16 | 0.003 |
| 390 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 49 → 65; т40: 16 → 0 | 0.003 |
| 391 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т40: 0 → 54; т41: 54 → 0 | 0.137 |
| 392 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т10: 51 → 0 | 0.012 |
| 393 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т9: 102 → 153; т10: 51 → 0 | 0.004 |
| 394 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т39: 29 → 101; т40: 145 → 72 | 0.000 |
| 395 | half+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т40: 72 → 36; т41: 0 → 36 | 0.000 |
| 396 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т29: 0 → 38; т30: 38 → 0 | 0.042 |
| 397 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т20: 37 → 0 | 0.004 |
| 398 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т33: 34 → 51 | 0.032 |
| 399 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т32: 22 → 73; т33: 51 → 0 | 0.000 |
| 400 | drop | valve | crude | `tg.tb.term_tw.grid_tw` | т15: 33 → 0 | 0.048 |
| 401 | half | valve | crude | `tg.tb.term_tw.grid_tw` | т7: 33 → 16 | 0.000 |
| 402 | drop | valve | crude | `tg.tb.term_tw.grid_tw` | т9: 33 → 0 | 0.001 |
| 403 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т36: 33 → 0 | 0.000 |
| 404 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т37: 33 → 16 | 0.000 |
| 405 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т37: 16 → 8 | 0.000 |
| 406 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т37: 8 → 4 | 0.000 |
| 407 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т48: 33 → 0 | 0.000 |
| 408 | shift+1 | valve | crude | `tg.tb.term_tw.grid_tw` | т50: 33 → 0; т51: 33 → 65 | 0.000 |
| 409 | shift+1 | valve | crude | `tg.tb.term_tw.grid_tw` | т51: 65 → 0; т52: 33 → 98 | 0.000 |
| 410 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т34: 30 → 0; т35: 0 → 30 | 0.000 |
| 411 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т31: 0 → 73; т32: 73 → 0 | 0.000 |
| 412 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т52: 160 → 241 | 0.006 |
| 413 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т31: 19 → 0; т32: 10 → 29 | 0.000 |
| 414 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 18 → 27 | 0.027 |
| 415 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 27 → 13; т3: 0 → 13 | 0.015 |
| 416 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 13 → 7; т3: 13 → 20 | 0.007 |
| 417 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т33: 16 → 0; т34: 0 → 16 | 0.000 |
| 418 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т32: 29 → 0; т33: 0 → 29 | 0.000 |
| 419 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 9 → 14 | 0.019 |
| 420 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 14 → 21 | 0.028 |
| 421 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 21 → 31 | 0.042 |
| 422 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 9 → 4; т2: 7 → 11 | 0.003 |
| 423 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 4 → 2; т2: 11 → 13 | 0.001 |
| 424 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т19: 0 → 6,075; т20: 12,150 → 6,075 | 0.000 |
| 425 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 6,075 → 9,112 | 0.000 |
| 426 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 9,112 → 13,669 | 0.000 |
| 427 | shift-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т51: 5,898 → 14,781; т52: 8,882 → 0 | 0.000 |
| 428 | half+1 | order | lng | `sea.tb.src_us_lng.term_eu` | т46: 7,940 → 3,970; т47: 3,970 → 7,940 | 0.000 |
| 429 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т12: 7,523 → 11,285 | 0.001 |
| 430 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т33: 0 → 3,240; т34: 6,480 → 3,240 | 0.000 |
| 431 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 3,240 → 4,860 | 0.000 |
| 432 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 4,860 → 7,290 | 0.000 |
| 433 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т25: 5,400 → 0; т26: 0 → 5,400 | 0.000 |
| 434 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 5,051 → 0; т24: 0 → 5,051 | 0.000 |
| 435 | half | valve | lng | `tg.tb.term_tw.grid_tw` | т2: 4,243 → 2,121 | 0.031 |
| 436 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т2: 2,121 → 3,182 | 0.260 |
| 437 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т2: 3,182 → 1,591; т3: 0 → 1,591 | 0.590 |
| 438 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т7: 0 → 2,029; т8: 4,059 → 2,029 | 0.077 |
| 439 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т8: 2,029 → 1,015; т9: 0 → 1,015 | 0.010 |
| 440 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т8: 1,015 → 1,522 | 0.017 |
| 441 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т21: 0 → 1,890; т22: 3,780 → 1,890 | 0.000 |
| 442 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 1,890 → 945; т23: 0 → 945 | 0.000 |
| 443 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 945 → 1,417 | 0.000 |
| 444 | shift+2 | valve | lng | `tg.tb.term_kr.grid_kr` | т16: 2,750 → 0; т18: 0 → 2,750 | 0.000 |
| 445 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т17: 0 → 1,353; т18: 2,706 → 1,353 | 0.419 |
| 446 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т48: 0 → 1,125; т49: 2,250 → 1,125 | 0.000 |
| 447 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т44: 1,938 → 969; т45: 0 → 969 | 0.000 |
| 448 | shift-2 | valve | lng | `tg.tb.term_kr.grid_kr` | т9: 0 → 1,925; т11: 1,925 → 0 | 0.000 |
| 449 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т41: 1,900 → 0; т42: 0 → 1,900 | 0.000 |
| 450 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 0 → 2,025; т23: 2,025 → 0 | 0.000 |
| 451 | half-1 | valve | lng | `tg.tb.term_tw.grid_tw` | т19: 762 → 1,619; т20: 1,714 → 857 | 0.000 |
| 452 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т20: 857 → 1,286 | 0.000 |
| 453 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т20: 1,286 → 1,929 | 0.000 |
| 454 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 1,614 → 807; т38: 1,046 → 1,853 | 0.000 |
| 455 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 807 → 403; т38: 1,853 → 2,256 | 0.000 |
| 456 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 403 → 0; т39: 607 → 1,011 | 0.000 |
| 457 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т46: 0 → 1,595; т47: 1,595 → 0 | 0.000 |
| 458 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т31: 0 → 1,591; т32: 1,591 → 0 | 0.000 |
| 459 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т24: 0 → 1,694; т25: 1,694 → 0 | 0.000 |
| 460 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 1,557 → 0; т19: 519 → 2,076 | 0.000 |
| 461 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 0 → 773; т44: 1,545 → 773 | 0.000 |
| 462 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т41: 1,451 → 0; т42: 0 → 1,451 | 0.000 |
| 463 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 1,430 → 715; т6: 953 → 1,668 | 0.078 |
| 464 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т25: 0 → 759; т26: 1,519 → 759 | 0.000 |
| 465 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т26: 1,284 → 642; т27: 0 → 642 | 0.000 |
| 466 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т26: 642 → 963 | 0.000 |
| 467 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т30: 1,281 → 0; т31: 383 → 1,664 | 0.000 |
| 468 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т27: 0 → 686; т28: 1,371 → 686 | 0.000 |
| 469 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 686 → 1,028 | 0.000 |
| 470 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т19: 0 → 675; т20: 1,350 → 675 | 0.000 |
| 471 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т2: 1,253 → 0; т4: 0 → 1,253 | 0.001 |
| 472 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т18: 1,240 → 620 | 0.006 |
| 473 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т18: 620 → 0; т19: 0 → 620 | 0.000 |
| 474 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т7: 0 → 1,320; т8: 1,320 → 0 | 0.000 |
| 475 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т6: 1,215 → 0; т7: 0 → 1,215 | 0.000 |
| 476 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 1,177 → 589; т25: 392 → 981 | 0.000 |
| 477 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т26: 1,177 → 589; т27: 392 → 981 | 0.000 |
| 478 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т7: 1,248 → 0; т9: 0 → 1,248 | 0.000 |
| 479 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т40: 1,248 → 0; т42: 0 → 1,248 | 0.000 |
| 480 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т33: 1,156 → 0; т34: 849 → 2,005 | 0.000 |
| 481 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т13: 1,225 → 1,838 | 0.000 |
| 482 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т43: 1,147 → 0; т44: 0 → 1,147 | 0.000 |
| 483 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т8: 1,038 → 1,557; т9: 1,038 → 519 | 0.001 |
| 484 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 519 → 0; т10: 987 → 1,506 | 0.000 |
| 485 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т17: 1,038 → 0; т18: 0 → 1,038 | 0.000 |
| 486 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т40: 1,027 → 0; т41: 0 → 1,027 | 0.008 |
| 487 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т20: 1,001 → 0; т21: 0 → 1,001 | 0.000 |
| 488 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т17: 949 → 475; т18: 0 → 475 | 0.000 |
| 489 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 450 → 956; т36: 1,012 → 506 | 0.000 |
| 490 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 506 → 759 | 0.000 |
| 491 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т20: 894 → 0; т21: 68 → 962 | 0.000 |
| 492 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т41: 877 → 1,315 | 0.000 |
| 493 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т38: 868 → 1,302 | 0.000 |
| 494 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т39: 858 → 1,287 | 0.000 |
| 495 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т41: 855 → 427; т42: 427 → 855 | 0.000 |
| 496 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т34: 2,005 → 0; т35: 0 → 2,005 | 0.001 |
| 497 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т15: 838 → 1,257 | 0.000 |
| 498 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т15: 1,257 → 1,885 | 0.000 |
| 499 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т40: 797 → 1,196 | 0.000 |
| 500 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т12: 794 → 0; т13: 0 → 794 | 0.000 |
| 501 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т32: 837 → 1,256 | 0.000 |
| 502 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 506 → 1,134; т32: 1,256 → 628 | 0.000 |
| 503 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т32: 628 → 942 | 0.000 |
| 504 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т45: 828 → 0; т46: 212 → 1,040 | 0.000 |
| 505 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т32: 770 → 0; т33: 0 → 770 | 0.000 |
| 506 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т41: 810 → 405; т42: 135 → 540 | 0.000 |
| 507 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т4: 712 → 0; т5: 475 → 1,187 | 0.000 |
| 508 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т19: 675 → 1,030; т20: 711 → 356 | 0.011 |
| 509 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т20: 356 → 178; т21: 149 → 327 | 0.000 |
| 510 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т20: 178 → 89; т21: 327 → 416 | 0.000 |
| 511 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т14: 706 → 0; т15: 706 → 1,412 | 0.000 |
| 512 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т15: 1,412 → 706; т16: 706 → 1,412 | 0.000 |
| 513 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т15: 706 → 353; т16: 1,412 → 1,765 | 0.000 |
| 514 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т16: 1,765 → 882; т17: 706 → 1,588 | 0.000 |
| 515 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т17: 1,588 → 794; т18: 0 → 794 | 0.000 |
| 516 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т17: 794 → 397; т18: 794 → 1,191 | 0.000 |
| 517 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 919 → 1,435; т19: 1,030 → 515 | 0.176 |
| 518 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т19: 515 → 0; т20: 89 → 604 | 0.000 |
| 519 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т8: 667 → 334 | 0.001 |
| 520 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т8: 334 → 167 | 0.000 |
| 521 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т10: 667 → 334 | 0.001 |
| 522 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т12: 667 → 334 | 0.001 |
| 523 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т12: 334 → 167 | 0.000 |
| 524 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т19: 667 → 0; т21: 0 → 667 | 0.000 |
| 525 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т45: 667 → 0; т47: 0 → 667 | 0.000 |
| 526 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т39: 1,011 → 0; т40: 0 → 1,011 | 0.000 |
| 527 | shift-2 | valve | crude | `tg.tb.term_jp.grid_jp` | т10: 0 → 589; т12: 589 → 0 | 0.000 |
| 528 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т19: 2,076 → 0; т20: 0 → 2,076 | 0.001 |
| 529 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т9: 547 → 273 | 0.003 |
| 530 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 506 → 759 | 0.000 |
| 531 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т23: 490 → 245; т24: 0 → 245 | 0.026 |
| 532 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 486 → 243; т14: 486 → 729 | 0.000 |
| 533 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т15: 486 → 243; т16: 486 → 729 | 0.000 |
| 534 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т25: 486 → 972; т26: 486 → 0 | 0.000 |
| 535 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 262 → 728; т19: 466 → 0 | 0.027 |
| 536 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т43: 427 → 214; т44: 0 → 214 | 0.000 |
| 537 | shift+2 | valve | crude | `tg.tb.term_kr.grid_kr` | т45: 440 → 0; т47: 206 → 646 | 0.000 |
| 538 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т31: 1,664 → 0; т32: 0 → 1,664 | 0.000 |
| 539 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т16: 408 → 612 | 0.000 |
| 540 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т23: 405 → 202; т24: 135 → 338 | 0.000 |
| 541 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т27: 405 → 202; т28: 135 → 338 | 0.000 |
| 542 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т29: 405 → 202; т30: 135 → 338 | 0.000 |
| 543 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т31: 405 → 202; т32: 135 → 338 | 0.000 |
| 544 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т33: 405 → 202; т34: 135 → 338 | 0.000 |
| 545 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т35: 405 → 202; т36: 135 → 338 | 0.000 |
| 546 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 338 → 742; т37: 405 → 0 | 0.000 |
| 547 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т9: 0 → 355; т10: 355 → 0 | 0.000 |
| 548 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т48: 347 → 0; т49: 0 → 347 | 0.000 |
| 549 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т27: 300 → 150 | 0.000 |
| 550 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т43: 300 → 0; т45: 0 → 300 | 0.000 |
| 551 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 293 → 439 | 0.000 |
| 552 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 439 → 659 | 0.000 |
| 553 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 659 → 988 | 0.000 |
| 554 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 263 → 394 | 0.000 |
| 555 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т21: 394 → 592 | 0.000 |
| 556 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 728 → 364; т19: 0 → 364 | 0.000 |
| 557 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 364 → 0; т20: 0 → 364 | 0.000 |
| 558 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т46: 1,040 → 0; т47: 0 → 1,040 | 0.000 |
| 559 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т5: 165 → 0 | 0.002 |
| 560 | half-1 | valve | crude | `tg.tb.term_tw.grid_tw` | т19: 0 → 82; т20: 163 → 82 | 0.000 |
| 561 | half-1 | valve | crude | `tg.tb.term_tw.grid_tw` | т19: 82 → 122; т20: 82 → 41 | 0.000 |
| 562 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т43: 153 → 229 | 0.150 |
| 563 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т43: 229 → 0; т44: 0 → 229 | 0.000 |
| 564 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т45: 0 → 69; т46: 137 → 69 | 0.000 |
| 565 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т22: 135 → 68; т23: 202 → 270 | 0.000 |
| 566 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т26: 135 → 68; т27: 202 → 270 | 0.000 |
| 567 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т25: 405 → 439; т26: 68 → 34 | 0.000 |
| 568 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т37: 0 → 135; т38: 135 → 0 | 0.000 |
| 569 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 540 → 270; т43: 0 → 270 | 0.000 |
| 570 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 270 → 135; т43: 270 → 405 | 0.000 |
| 571 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 130 → 0; т8: 95 → 224 | 0.018 |
| 572 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т19: 110 → 0; т20: 0 → 110 | 0.000 |
| 573 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 0 → 112; т8: 224 → 112 | 0.000 |
| 574 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т2: 83 → 0; т3: 0 → 83 | 0.000 |
| 575 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т16: 76 → 114 | 0.000 |
| 576 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т30: 0 → 73; т31: 73 → 0 | 0.000 |
| 577 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 962 → 0; т22: 0 → 962 | 0.000 |
| 578 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 65 → 33; т39: 494 → 526 | 0.000 |
| 579 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 33 → 16; т39: 526 → 543 | 0.000 |
| 580 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т38: 16 → 0; т40: 54 → 70 | 0.000 |
| 581 | half | valve | crude | `tg.tb.term_tw.grid_tw` | т3: 54 → 27 | 0.018 |
| 582 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т1: 54 → 0 | 0.001 |
| 583 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т3: 47 → 24; т4: 425 → 449 | 0.004 |
| 584 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т29: 38 → 57 | 0.039 |
| 585 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т29: 57 → 86 | 0.079 |
| 586 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т28: 0 → 43; т29: 86 → 43 | 0.007 |
| 587 | shift-1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т37: 4 → 37; т38: 33 → 0 | 0.000 |
| 588 | half | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т39: 33 → 16 | 0.000 |
| 589 | drop | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т49: 33 → 0 | 0.000 |
| 590 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т37: 32 → 48 | 0.000 |
| 591 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 31 → 0 | 0.000 |
| 592 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т35: 30 → 0 | 0.000 |
| 593 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т33: 29 → 0; т34: 16 → 45 | 0.000 |
| 594 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 13 → 23; т3: 20 → 10 | 0.001 |
| 595 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т34: 45 → 0; т35: 0 → 45 | 0.000 |
| 596 | shift+1 | order | crude | `sea.tb.src_us_crude.chk_panama [lane.src_us_crude.term_tw]` | т37: 37 → 0; т38: 0 → 37 | 0.000 |
| 597 | half+1 | order | lng | `sea.tb.src_us_lng.term_eu` | т45: 7,940 → 3,970; т46: 3,970 → 7,940 | 0.000 |
| 598 | shift-2 | valve | lng | `tg.tb.term_jp.grid_jp` | т32: 0 → 7,290; т34: 7,290 → 0 | 0.000 |
| 599 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т18: 0 → 6,075; т19: 6,075 → 0 | 0.000 |
| 600 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т26: 5,400 → 0; т27: 0 → 5,400 | 0.000 |
| 601 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т24: 5,051 → 0; т25: 0 → 5,051 | 0.000 |
| 602 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 0 → 2,025; т35: 4,050 → 2,025 | 0.000 |
| 603 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т4: 3,891 → 1,946; т5: 0 → 1,946 | 0.000 |
| 604 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т4: 1,946 → 2,918 | 0.000 |
| 605 | shift-2 | valve | lng | `tg.tb.term_jp.grid_jp` | т31: 0 → 3,240; т33: 3,240 → 0 | 0.000 |
| 606 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т15: 2,750 → 0; т16: 0 → 2,750 | 0.000 |
| 607 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т20: 2,076 → 0; т21: 1,001 → 3,077 | 0.001 |
| 608 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т35: 2,005 → 1,003; т36: 0 → 1,003 | 0.000 |
| 609 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т8: 0 → 1,925; т9: 1,925 → 0 | 0.000 |
| 610 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т40: 1,900 → 950; т41: 0 → 950 | 0.000 |

## Епізод 11: J 2809.06 → 2771.89 млрд USD (−37.17), прийнято 805 ходів

| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |
|---|---|---|---|---|---|---|
| 1 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т21: 8,907 → 13,361 | 0.000 |
| 2 | shift-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т51: 8,629 → 17,511; т52: 8,882 → 0 | 0.000 |
| 3 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т3: 8,849 → 13,273 | 0.000 |
| 4 | shift-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т2: 8,802 → 22,075; т3: 13,273 → 0 | 0.000 |
| 5 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т1: 1,670 → 12,707; т2: 22,075 → 11,038 | 0.000 |
| 6 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т39: 8,629 → 12,943 | 0.000 |
| 7 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т41: 8,629 → 12,943 | 0.000 |
| 8 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 0 → 8,100; т21: 8,100 → 0 | 1.347 |
| 9 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 0 → 6,480; т36: 6,480 → 0 | 0.376 |
| 10 | drop | order | lng | `sea.tb.src_us_lng.term_eu` | т49: 5,651 → 0 | 0.000 |
| 11 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т3: 5,423 → 8,134 | 0.000 |
| 12 | shift+2 | valve | lng | `tg.tb.term_jp.grid_jp` | т4: 5,400 → 0; т6: 0 → 5,400 | 0.000 |
| 13 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т1: 934 → 6,221; т2: 5,287 → 0 | 0.004 |
| 14 | shift+2 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 5,287 → 0; т25: 0 → 5,287 | 0.000 |
| 15 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т36: 0 → 2,642; т37: 5,284 → 2,642 | 0.000 |
| 16 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т36: 2,642 → 3,963; т37: 2,642 → 1,321 | 0.000 |
| 17 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 3,645 → 5,467 | 0.000 |
| 18 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т21: 0 → 2,734; т22: 5,467 → 2,734 | 0.000 |
| 19 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 2,734 → 4,100 | 0.000 |
| 20 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т18: 0 → 3,182; т19: 3,182 → 0 | 2.002 |
| 21 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т29: 0 → 3,182; т30: 3,182 → 0 | 0.940 |
| 22 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т8: 2,952 → 4,428 | 0.000 |
| 23 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т10: 2,699 → 0; т11: 2,647 → 5,346 | 0.902 |
| 24 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т14: 2,696 → 4,044 | 0.000 |
| 25 | shift+2 | valve | lng | `tg.tb.term_kr.grid_kr` | т14: 4,044 → 0; т16: 0 → 4,044 | 1.785 |
| 26 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т1: 1,010 → 3,703; т2: 2,693 → 0 | 0.027 |
| 27 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т3: 2,651 → 5,308; т4: 2,657 → 0 | 0.070 |
| 28 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т3: 5,308 → 0; т4: 0 → 5,308 | 1.040 |
| 29 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т11: 5,346 → 8,019 | 0.535 |
| 30 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т11: 8,019 → 0; т12: 2,640 → 10,659 | 0.953 |
| 31 | shift+2 | valve | lng | `tg.tb.term_kr.grid_kr` | т25: 2,640 → 0; т27: 0 → 2,640 | 1.053 |
| 32 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т35: 2,629 → 3,943 | 0.004 |
| 33 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т35: 3,943 → 1,971 | 0.292 |
| 34 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т33: 2,625 → 3,938; т34: 2,625 → 1,313 | 0.098 |
| 35 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т24: 2,493 → 3,740 | 0.000 |
| 36 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 0 → 3,740; т24: 3,740 → 0 | 0.000 |
| 37 | shift-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т8: 0 → 2,357; т9: 2,357 → 0 | 0.000 |
| 38 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т51: 2,351 → 3,527 | 0.000 |
| 39 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т43: 2,316 → 3,473 | 0.000 |
| 40 | x1.5 | order | lng | `sea.tb.src_us_lng.term_eu` | т17: 1,985 → 2,977 | 2.935 |
| 41 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т5: 1,801 → 2,701 | 0.000 |
| 42 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т5: 2,701 → 0; т6: 0 → 2,701 | 0.119 |
| 43 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т15: 1,643 → 2,465 | 0.000 |
| 44 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т45: 1,330 → 1,995 | 0.000 |
| 45 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 1,350 → 2,025 | 0.000 |
| 46 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т35: 189 → 2,214; т36: 2,025 → 0 | 0.000 |
| 47 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 1,349 → 2,024 | 0.000 |
| 48 | half | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 2,024 → 1,012 | 0.702 |
| 49 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т5: 1,217 → 1,826 | 0.000 |
| 50 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т5: 1,826 → 2,739 | 0.000 |
| 51 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т5: 2,739 → 4,108 | 0.000 |
| 52 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т1: 1,169 → 1,753 | 0.096 |
| 53 | half | order | crude | `sea.tb.src_us_crude.term_eu` | т20: 1,248 → 624 | 0.001 |
| 54 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т43: 1,248 → 0; т45: 104 → 1,352 | 0.000 |
| 55 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т44: 1,248 → 0; т46: 0 → 1,248 | 0.000 |
| 56 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т32: 1,075 → 538 | 0.009 |
| 57 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т31: 789 → 1,326; т32: 538 → 0 | 0.051 |
| 58 | half | valve | crude | `tg.tb.term_eu.grid_eu` | т23: 1,144 → 572 | 0.000 |
| 59 | x2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 1,032 → 2,063 | 0.163 |
| 60 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 2,063 → 3,095 | 0.015 |
| 61 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т33: 1,031 → 0 | 0.545 |
| 62 | shift+1 | order | lng | `sea.tb.src_ru_gas.term_jp` | т26: 1,028 → 0; т27: 72 → 1,100 | 2.036 |
| 63 | drop | order | lng | `sea.tb.src_ru_gas.term_jp` | т50: 1,028 → 0 | 0.004 |
| 64 | drop | order | lng | `sea.tb.src_ru_gas.term_jp` | т51: 1,028 → 0 | 0.004 |
| 65 | drop | order | lng | `sea.tb.src_ru_gas.term_jp` | т52: 1,028 → 0 | 0.004 |
| 66 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т36: 1,010 → 0 | 0.041 |
| 67 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т1: 3,703 → 5,555 | 0.000 |
| 68 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т34: 1,007 → 0; т35: 999 → 2,006 | 0.000 |
| 69 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т35: 2,006 → 1,003; т36: 0 → 1,003 | 0.000 |
| 70 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т35: 1,003 → 1,505 | 0.149 |
| 71 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т35: 1,505 → 752; т36: 1,003 → 1,756 | 0.004 |
| 72 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т1: 1,041 → 1,562 | 0.000 |
| 73 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т4: 1,040 → 1,560 | 0.000 |
| 74 | shift+1 | valve | crude | `tg.tb.term_eu.grid_eu` | т4: 1,560 → 0; т5: 0 → 1,560 | 0.000 |
| 75 | half-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т25: 1,040 → 1,560; т26: 1,040 → 520 | 0.006 |
| 76 | half-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т26: 520 → 1,040; т27: 1,040 → 520 | 0.001 |
| 77 | half-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т26: 1,040 → 1,300; т27: 520 → 260 | 0.000 |
| 78 | shift-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т19: 0 → 1,039; т20: 1,039 → 0 | 0.000 |
| 79 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т8: 930 → 1,899; т10: 968 → 0 | 0.048 |
| 80 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т20: 965 → 482; т21: 873 → 1,355 | 0.000 |
| 81 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т2: 963 → 481; т3: 652 → 1,134 | 0.133 |
| 82 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 936 → 1,404 | 0.010 |
| 83 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 1,404 → 702; т10: 0 → 702 | 0.019 |
| 84 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 702 → 1,053 | 0.027 |
| 85 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т1: 6,221 → 3,111; т2: 0 → 3,111 | 0.000 |
| 86 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т41: 931 → 1,397 | 0.000 |
| 87 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т41: 1,397 → 2,095 | 0.005 |
| 88 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т42: 931 → 466; т43: 931 → 1,397 | 0.000 |
| 89 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т37: 931 → 0; т38: 931 → 1,862 | 0.000 |
| 90 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т44: 931 → 0 | 0.001 |
| 91 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т45: 931 → 0 | 0.001 |
| 92 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т46: 931 → 0 | 0.001 |
| 93 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т7: 900 → 2,798; т8: 1,899 → 0 | 0.001 |
| 94 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т19: 929 → 0; т20: 482 → 1,411 | 0.000 |
| 95 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т17: 909 → 454 | 0.075 |
| 96 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т17: 454 → 227; т18: 857 → 1,084 | 0.000 |
| 97 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т17: 227 → 114; т18: 1,084 → 1,197 | 0.000 |
| 98 | half-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т6: 798 → 2,198; т7: 2,798 → 1,399 | 0.046 |
| 99 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т10: 702 → 1,577; т11: 875 → 0 | 0.020 |
| 100 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т21: 1,355 → 678; т22: 827 → 1,504 | 0.000 |
| 101 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т47: 855 → 0 | 0.000 |
| 102 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т48: 855 → 0 | 0.000 |
| 103 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 900 → 1,350 | 0.000 |
| 104 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 1,350 → 2,025 | 0.000 |
| 105 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т3: 2,025 → 0; т5: 0 → 2,025 | 0.000 |
| 106 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 900 → 1,350 | 0.337 |
| 107 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т23: 900 → 0; т25: 0 → 900 | 0.375 |
| 108 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т24: 900 → 1,350 | 0.002 |
| 109 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т23: 0 → 1,350; т24: 1,350 → 0 | 0.000 |
| 110 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 0 → 450; т37: 900 → 450 | 0.000 |
| 111 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т12: 836 → 0; т13: 808 → 1,644 | 0.000 |
| 112 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т16: 817 → 0; т17: 114 → 930 | 0.000 |
| 113 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т13: 1,644 → 822; т14: 795 → 1,617 | 0.000 |
| 114 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т13: 822 → 411; т14: 1,617 → 2,028 | 0.000 |
| 115 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т47: 803 → 0 | 0.001 |
| 116 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т4: 800 → 1,200 | 0.000 |
| 117 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т4: 1,200 → 600; т5: 771 → 1,371 | 0.000 |
| 118 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т4: 600 → 300 | 0.005 |
| 119 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т14: 2,028 → 1,014; т15: 788 → 1,801 | 0.000 |
| 120 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т24: 792 → 0; т26: 784 → 1,576 | 0.033 |
| 121 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т31: 1,326 → 0; т32: 0 → 1,326 | 0.000 |
| 122 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т15: 1,801 → 901; т16: 0 → 901 | 0.000 |
| 123 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т15: 901 → 450; т16: 901 → 1,351 | 0.000 |
| 124 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т24: 0 → 787; т25: 787 → 0 | 0.003 |
| 125 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т30: 785 → 393; т31: 0 → 393 | 0.000 |
| 126 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т30: 393 → 196; т31: 393 → 589 | 0.000 |
| 127 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т29: 781 → 977; т30: 196 → 0 | 0.006 |
| 128 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т28: 780 → 0; т29: 977 → 1,757 | 0.115 |
| 129 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw.lombok]` | т49: 756 → 0 | 0.000 |
| 130 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т47: 706 → 0 | 0.001 |
| 131 | x1.5 | order | lng | `sea.tb.src_au_lng.term_kr` | т48: 697 → 1,045 | 0.000 |
| 132 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т19: 667 → 334 | 0.002 |
| 133 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т21: 667 → 334 | 0.001 |
| 134 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т23: 667 → 334 | 0.010 |
| 135 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т25: 667 → 334 | 0.004 |
| 136 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т27: 667 → 334 | 0.009 |
| 137 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т29: 667 → 334 | 0.007 |
| 138 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т31: 667 → 334 | 0.003 |
| 139 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т33: 667 → 334 | 0.000 |
| 140 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т34: 667 → 334 | 0.061 |
| 141 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т38: 667 → 334 | 0.018 |
| 142 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т40: 667 → 334 | 0.000 |
| 143 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т40: 334 → 167 | 0.000 |
| 144 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т41: 667 → 334 | 0.000 |
| 145 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т43: 667 → 334 | 0.000 |
| 146 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т43: 334 → 167 | 0.000 |
| 147 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т44: 667 → 334 | 0.000 |
| 148 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т44: 334 → 1,001; т45: 667 → 0 | 0.003 |
| 149 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т46: 667 → 334 | 0.002 |
| 150 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т46: 334 → 167 | 0.001 |
| 151 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т44: 1,001 → 1,167; т46: 167 → 0 | 0.001 |
| 152 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т48: 667 → 0 | 0.003 |
| 153 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т3: 1,134 → 567; т4: 300 → 867 | 0.000 |
| 154 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т3: 567 → 850 | 0.072 |
| 155 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 638 → 0 | 0.002 |
| 156 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т44: 638 → 0 | 0.001 |
| 157 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т45: 638 → 0 | 0.001 |
| 158 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т46: 638 → 0 | 0.001 |
| 159 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 635 → 0; т43: 0 → 635 | 0.000 |
| 160 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т41: 619 → 0; т42: 0 → 619 | 0.000 |
| 161 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т40: 601 → 300; т41: 0 → 300 | 0.000 |
| 162 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т40: 300 → 150; т41: 300 → 450 | 0.000 |
| 163 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т39: 592 → 296; т40: 150 → 446 | 0.000 |
| 164 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 453 → 749; т39: 296 → 0 | 0.001 |
| 165 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т42: 554 → 830 | 0.000 |
| 166 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т35: 475 → 0 | 0.004 |
| 167 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 475 → 0 | 0.004 |
| 168 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т37: 475 → 0 | 0.002 |
| 169 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т38: 475 → 0 | 0.002 |
| 170 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т39: 475 → 0 | 0.002 |
| 171 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т40: 475 → 237 | 0.001 |
| 172 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т40: 237 → 119 | 0.000 |
| 173 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т48: 475 → 237 | 0.000 |
| 174 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т48: 237 → 119 | 0.000 |
| 175 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т48: 119 → 59 | 0.000 |
| 176 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т49: 475 → 0 | 0.000 |
| 177 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 749 → 1,123 | 0.004 |
| 178 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 1,123 → 1,685 | 0.052 |
| 179 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 1,685 → 2,527 | 0.265 |
| 180 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т38: 436 → 654 | 0.016 |
| 181 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т38: 654 → 327; т39: 0 → 327 | 0.000 |
| 182 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т35: 366 → 0; т36: 366 → 733 | 0.000 |
| 183 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т18: 392 → 587 | 0.000 |
| 184 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т18: 587 → 881 | 0.000 |
| 185 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т18: 881 → 1,321 | 0.000 |
| 186 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т12: 392 → 587 | 0.000 |
| 187 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т12: 587 → 881 | 0.000 |
| 188 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т12: 881 → 1,321 | 0.000 |
| 189 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т13: 392 → 587 | 0.000 |
| 190 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т13: 587 → 881 | 0.000 |
| 191 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т13: 881 → 1,321 | 0.000 |
| 192 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т15: 392 → 587 | 0.000 |
| 193 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т32: 392 → 587 | 0.000 |
| 194 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т32: 587 → 881 | 0.000 |
| 195 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т32: 881 → 1,321 | 0.000 |
| 196 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 383 → 574 | 0.000 |
| 197 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 574 → 861 | 0.000 |
| 198 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т31: 861 → 1,292 | 0.000 |
| 199 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т2: 38 → 387; т3: 349 → 0 | 0.020 |
| 200 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т37: 330 → 495 | 0.076 |
| 201 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т37: 495 → 0; т38: 157 → 652 | 0.978 |
| 202 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т1: 329 → 494 | 0.000 |
| 203 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т33: 300 → 150 | 0.048 |
| 204 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т44: 300 → 600; т45: 300 → 0 | 0.001 |
| 205 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т46: 300 → 0 | 0.002 |
| 206 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т47: 300 → 0 | 0.002 |
| 207 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т48: 300 → 0 | 0.002 |
| 208 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т30: 316 → 474 | 0.000 |
| 209 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т30: 474 → 711 | 0.000 |
| 210 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т30: 711 → 1,066 | 0.000 |
| 211 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 270 → 0 | 0.348 |
| 212 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т2: 0 → 135; т3: 270 → 135 | 0.000 |
| 213 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т3: 135 → 202 | 0.012 |
| 214 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т4: 270 → 135; т5: 270 → 405 | 0.000 |
| 215 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т7: 270 → 0 | 0.251 |
| 216 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т7: 0 → 270; т8: 270 → 0 | 0.550 |
| 217 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т8: 0 → 270; т9: 270 → 0 | 0.008 |
| 218 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т9: 0 → 135; т10: 270 → 135 | 0.011 |
| 219 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т10: 135 → 0 | 0.001 |
| 220 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т11: 270 → 0 | 0.002 |
| 221 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т23: 270 → 0; т24: 270 → 540 | 0.000 |
| 222 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т24: 540 → 0; т25: 270 → 810 | 0.000 |
| 223 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т26: 270 → 0; т28: 270 → 540 | 0.003 |
| 224 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т28: 540 → 0 | 0.029 |
| 225 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т29: 270 → 405 | 0.063 |
| 226 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т29: 405 → 0; т30: 270 → 675 | 0.000 |
| 227 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т30: 675 → 0 | 0.221 |
| 228 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т31: 270 → 0; т32: 68 → 338 | 0.000 |
| 229 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т22: 270 → 0 | 0.080 |
| 230 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т21: 270 → 405 | 0.030 |
| 231 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т21: 405 → 202; т22: 0 → 202 | 0.000 |
| 232 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т21: 202 → 304 | 0.022 |
| 233 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т20: 270 → 0; т21: 304 → 574 | 0.000 |
| 234 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т19: 270 → 135 | 0.001 |
| 235 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т19: 135 → 67 | 0.000 |
| 236 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т18: 270 → 0; т19: 67 → 337 | 0.000 |
| 237 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т17: 270 → 0 | 0.034 |
| 238 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т15: 269 → 403 | 0.484 |
| 239 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т13: 266 → 669; т15: 403 → 0 | 0.070 |
| 240 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т14: 268 → 402 | 0.449 |
| 241 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т14: 402 → 0; т15: 0 → 402 | 0.000 |
| 242 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 245 → 367 | 0.000 |
| 243 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 367 → 551 | 0.000 |
| 244 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 551 → 826 | 0.000 |
| 245 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т27: 238 → 357 | 0.052 |
| 246 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 230 → 115; т6: 202 → 317 | 0.000 |
| 247 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 115 → 57; т6: 317 → 375 | 0.000 |
| 248 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т5: 57 → 29; т6: 375 → 403 | 0.000 |
| 249 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т28: 220 → 330 | 0.070 |
| 250 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т28: 330 → 0; т29: 220 → 551 | 0.000 |
| 251 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т32: 217 → 327; т33: 220 → 110 | 0.000 |
| 252 | half+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т35: 220 → 110; т36: 215 → 325 | 0.029 |
| 253 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т28: 0 → 551; т29: 551 → 0 | 0.008 |
| 254 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т13: 220 → 0; т14: 181 → 401 | 0.000 |
| 255 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т36: 325 → 488 | 0.012 |
| 256 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т30: 215 → 0; т31: 212 → 427 | 0.000 |
| 257 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т15: 214 → 321 | 0.013 |
| 258 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т47: 210 → 0 | 0.000 |
| 259 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т24: 209 → 104; т25: 214 → 319 | 0.000 |
| 260 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т24: 104 → 52; т25: 319 → 371 | 0.000 |
| 261 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т24: 52 → 26; т25: 371 → 397 | 0.000 |
| 262 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т30: 209 → 0 | 0.127 |
| 263 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т38: 202 → 0 | 0.001 |
| 264 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т13: 197 → 98; т14: 208 → 307 | 0.000 |
| 265 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т13: 98 → 147 | 0.000 |
| 266 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т17: 196 → 0; т18: 0 → 196 | 0.002 |
| 267 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т34: 196 → 98; т35: 204 → 302 | 0.000 |
| 268 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т34: 98 → 49; т35: 302 → 351 | 0.000 |
| 269 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т34: 49 → 25; т35: 351 → 376 | 0.004 |
| 270 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т3: 176 → 0 | 0.003 |
| 271 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т4: 176 → 0; т5: 176 → 353 | 0.000 |
| 272 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т5: 353 → 0; т6: 176 → 529 | 0.000 |
| 273 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т6: 529 → 0; т7: 176 → 706 | 0.000 |
| 274 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т7: 706 → 0; т8: 176 → 882 | 0.000 |
| 275 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т8: 882 → 0; т9: 176 → 1,059 | 0.000 |
| 276 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т9: 1,059 → 0; т10: 176 → 1,235 | 0.000 |
| 277 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т10: 1,235 → 0; т11: 176 → 1,412 | 0.000 |
| 278 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т11: 1,412 → 0; т12: 176 → 1,588 | 0.000 |
| 279 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т12: 1,588 → 0; т13: 176 → 1,765 | 0.000 |
| 280 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т13: 1,765 → 0; т14: 176 → 1,941 | 0.000 |
| 281 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т14: 1,941 → 0; т15: 176 → 2,118 | 0.001 |
| 282 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т15: 2,118 → 0; т16: 176 → 2,294 | 0.001 |
| 283 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т16: 2,294 → 0; т17: 176 → 2,471 | 0.001 |
| 284 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т17: 2,471 → 0; т18: 176 → 2,647 | 0.001 |
| 285 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т18: 2,647 → 0; т19: 176 → 2,824 | 0.001 |
| 286 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т19: 2,824 → 1,412; т20: 176 → 1,588 | 0.000 |
| 287 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т19: 1,412 → 706; т20: 1,588 → 2,294 | 0.000 |
| 288 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т19: 706 → 0; т21: 176 → 882 | 0.000 |
| 289 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т20: 2,294 → 0; т21: 882 → 3,177 | 0.001 |
| 290 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 3,177 → 1,588; т22: 176 → 1,765 | 0.000 |
| 291 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т21: 1,588 → 0; т23: 176 → 1,765 | 0.001 |
| 292 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 1,765 → 882; т23: 1,765 → 2,647 | 0.000 |
| 293 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 882 → 441; т23: 2,647 → 3,089 | 0.000 |
| 294 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 441 → 221; т23: 3,089 → 3,309 | 0.000 |
| 295 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т23: 3,309 → 1,655; т24: 176 → 1,831 | 0.000 |
| 296 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т23: 1,655 → 827; т24: 1,831 → 2,658 | 0.000 |
| 297 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т23: 827 → 414; т24: 2,658 → 3,072 | 0.000 |
| 298 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т24: 3,072 → 0; т25: 176 → 3,249 | 0.001 |
| 299 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т25: 3,249 → 1,624; т26: 176 → 1,801 | 0.000 |
| 300 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 1,801 → 900; т27: 176 → 1,077 | 0.000 |
| 301 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 900 → 0; т28: 176 → 1,077 | 0.000 |
| 302 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т27: 1,077 → 0; т28: 1,077 → 2,154 | 0.000 |
| 303 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т28: 2,154 → 0; т29: 176 → 2,330 | 0.001 |
| 304 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т29: 2,330 → 0; т30: 176 → 2,507 | 0.001 |
| 305 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т30: 2,507 → 0; т31: 176 → 2,683 | 0.001 |
| 306 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т31: 2,683 → 1,342; т32: 176 → 1,518 | 0.000 |
| 307 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т31: 1,342 → 671; т32: 1,518 → 2,189 | 0.000 |
| 308 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т31: 671 → 335; т32: 2,189 → 2,524 | 0.000 |
| 309 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т32: 2,524 → 0; т33: 176 → 2,701 | 0.001 |
| 310 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 2,701 → 1,350; т34: 176 → 1,527 | 0.000 |
| 311 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 1,350 → 675; т34: 1,527 → 2,202 | 0.000 |
| 312 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 675 → 338; т34: 2,202 → 2,540 | 0.000 |
| 313 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 2,540 → 1,270; т35: 0 → 1,270 | 0.000 |
| 314 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 1,270 → 635; т35: 1,270 → 1,905 | 0.000 |
| 315 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т37: 185 → 0; т38: 0 → 185 | 0.000 |
| 316 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т14: 401 → 0; т15: 84 → 485 | 0.010 |
| 317 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т5: 177 → 0 | 0.050 |
| 318 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т2: 176 → 0; т3: 0 → 176 | 0.000 |
| 319 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т22: 174 → 0; т23: 197 → 371 | 0.000 |
| 320 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т12: 173 → 87; т13: 147 → 234 | 0.000 |
| 321 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т33: 173 → 0; т34: 25 → 197 | 0.000 |
| 322 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т42: 166 → 249 | 0.000 |
| 323 | half+1 | valve | crude | `tg.tb.term_tw.grid_tw` | т1: 163 → 81; т2: 108 → 189 | 0.009 |
| 324 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т18: 144 → 0; т19: 72 → 217 | 0.214 |
| 325 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т32: 128 → 192 | 0.035 |
| 326 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т32: 192 → 288 | 0.062 |
| 327 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т32: 288 → 0; т33: 0 → 288 | 0.000 |
| 328 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 128 → 192 | 0.041 |
| 329 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т21: 192 → 0; т22: 0 → 192 | 0.000 |
| 330 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т1: 119 → 0; т2: 119 → 237 | 0.000 |
| 331 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т2: 237 → 0; т3: 119 → 356 | 0.000 |
| 332 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т3: 356 → 0; т4: 119 → 475 | 0.000 |
| 333 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т4: 475 → 0; т5: 119 → 594 | 0.000 |
| 334 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т5: 594 → 0; т6: 119 → 712 | 0.000 |
| 335 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т6: 712 → 0; т7: 119 → 831 | 0.000 |
| 336 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т7: 831 → 0; т8: 119 → 950 | 0.000 |
| 337 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т8: 950 → 0; т9: 119 → 1,069 | 0.000 |
| 338 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т9: 1,069 → 0; т10: 119 → 1,187 | 0.000 |
| 339 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т10: 1,187 → 0; т11: 119 → 1,306 | 0.000 |
| 340 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т11: 1,306 → 0; т12: 119 → 1,425 | 0.000 |
| 341 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т12: 1,425 → 0; т13: 119 → 1,544 | 0.000 |
| 342 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т13: 1,544 → 0; т14: 119 → 1,662 | 0.000 |
| 343 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т14: 1,662 → 0; т15: 119 → 1,781 | 0.000 |
| 344 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т15: 1,781 → 0; т16: 119 → 1,900 | 0.000 |
| 345 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т16: 1,900 → 0; т17: 119 → 2,019 | 0.001 |
| 346 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т17: 2,019 → 0; т18: 119 → 2,137 | 0.001 |
| 347 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т18: 2,137 → 0; т19: 119 → 2,256 | 0.001 |
| 348 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т19: 2,256 → 0; т20: 119 → 2,375 | 0.001 |
| 349 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т20: 2,375 → 0; т21: 119 → 2,494 | 0.001 |
| 350 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т21: 2,494 → 1,247; т22: 119 → 1,366 | 0.000 |
| 351 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т21: 1,247 → 623; т22: 1,366 → 1,989 | 0.000 |
| 352 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т21: 623 → 312; т22: 1,989 → 2,301 | 0.000 |
| 353 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т22: 2,301 → 0; т23: 119 → 2,419 | 0.001 |
| 354 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т23: 2,419 → 0; т24: 119 → 2,538 | 0.001 |
| 355 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т24: 2,538 → 0; т25: 119 → 2,657 | 0.001 |
| 356 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т25: 2,657 → 1,328; т26: 119 → 1,447 | 0.000 |
| 357 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т25: 1,328 → 664; т26: 1,447 → 2,111 | 0.000 |
| 358 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т25: 664 → 332; т26: 2,111 → 2,444 | 0.000 |
| 359 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т26: 2,444 → 0; т28: 119 → 2,562 | 0.001 |
| 360 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т27: 119 → 0; т28: 2,562 → 2,681 | 0.000 |
| 361 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т28: 2,681 → 1,341; т29: 119 → 1,459 | 0.000 |
| 362 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т28: 1,341 → 670; т29: 1,459 → 2,130 | 0.000 |
| 363 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т28: 670 → 335; т29: 2,130 → 2,465 | 0.000 |
| 364 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т29: 2,465 → 0; т30: 119 → 2,583 | 0.001 |
| 365 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т30: 2,583 → 0; т31: 119 → 2,702 | 0.001 |
| 366 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т31: 2,702 → 1,351; т32: 119 → 1,470 | 0.000 |
| 367 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т31: 1,351 → 676; т32: 1,470 → 2,145 | 0.000 |
| 368 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т31: 676 → 338; т32: 2,145 → 2,483 | 0.000 |
| 369 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т32: 2,483 → 0; т33: 119 → 2,602 | 0.001 |
| 370 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т33: 2,602 → 0; т34: 119 → 2,721 | 0.001 |
| 371 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т34: 2,721 → 1,360; т35: 0 → 1,360 | 0.000 |
| 372 | shift+2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т34: 1,360 → 0; т36: 0 → 1,360 | 0.001 |
| 373 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т11: 126 → 0; т12: 87 → 213 | 0.000 |
| 374 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т47: 116 → 0 | 0.000 |
| 375 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т1: 122 → 183 | 0.004 |
| 376 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т1: 183 → 0; т2: 0 → 183 | 0.000 |
| 377 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 122 → 182 | 0.250 |
| 378 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 182 → 273 | 0.375 |
| 379 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 273 → 410 | 0.563 |
| 380 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 122 → 182 | 0.250 |
| 381 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 182 → 273 | 0.375 |
| 382 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 273 → 410 | 0.563 |
| 383 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 122 → 182 | 0.011 |
| 384 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т6: 182 → 0; т7: 122 → 304 | 0.000 |
| 385 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т7: 304 → 0; т8: 122 → 425 | 0.000 |
| 386 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т8: 425 → 0; т9: 122 → 547 | 0.000 |
| 387 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т9: 547 → 0; т10: 122 → 668 | 0.000 |
| 388 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т10: 668 → 0; т11: 122 → 790 | 0.000 |
| 389 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т11: 790 → 395; т12: 122 → 516 | 0.000 |
| 390 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т11: 395 → 197; т12: 516 → 714 | 0.000 |
| 391 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т11: 197 → 99; т12: 714 → 813 | 0.000 |
| 392 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т12: 813 → 406; т13: 122 → 528 | 0.000 |
| 393 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т14: 122 → 0; т15: 122 → 243 | 0.000 |
| 394 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т15: 243 → 0; т16: 122 → 364 | 0.000 |
| 395 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т16: 364 → 0; т17: 122 → 486 | 0.000 |
| 396 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т17: 486 → 0; т18: 122 → 608 | 0.000 |
| 397 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т18: 608 → 0; т19: 122 → 729 | 0.000 |
| 398 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т19: 729 → 364; т20: 122 → 486 | 0.000 |
| 399 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т19: 364 → 182; т20: 486 → 668 | 0.000 |
| 400 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т19: 182 → 91; т20: 668 → 759 | 0.000 |
| 401 | x2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т20: 759 → 1,519 | 0.015 |
| 402 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т20: 1,519 → 2,278 | 0.012 |
| 403 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т21: 122 → 0 | 0.000 |
| 404 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т22: 122 → 0 | 0.000 |
| 405 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т23: 122 → 0 | 0.000 |
| 406 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т24: 122 → 0 | 0.000 |
| 407 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т25: 122 → 0; т26: 122 → 243 | 0.000 |
| 408 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т26: 243 → 0; т27: 9 → 252 | 0.000 |
| 409 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т28: 122 → 0; т29: 122 → 243 | 0.000 |
| 410 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т29: 243 → 0; т30: 122 → 364 | 0.000 |
| 411 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т30: 364 → 0; т31: 122 → 486 | 0.000 |
| 412 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т31: 486 → 0; т32: 122 → 608 | 0.000 |
| 413 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т32: 608 → 304; т33: 122 → 425 | 0.000 |
| 414 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т32: 304 → 152; т33: 425 → 577 | 0.000 |
| 415 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 577 → 289; т34: 122 → 410 | 0.000 |
| 416 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т52: 120 → 0 | 0.000 |
| 417 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т46: 120 → 0 | 0.000 |
| 418 | half-1 | order | crude | `sea.tb.src_us_crude.term_eu` | т44: 0 → 676; т45: 1,352 → 676 | 0.000 |
| 419 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т45: 676 → 0; т47: 0 → 676 | 0.000 |
| 420 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 102 → 0; т2: 135 → 237 | 0.000 |
| 421 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т7: 101 → 51; т8: 70 → 121 | 0.014 |
| 422 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т45: 87 → 0 | 0.000 |
| 423 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т45: 89 → 0 | 0.000 |
| 424 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 89 → 133 | 0.000 |
| 425 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 133 → 200 | 0.000 |
| 426 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 200 → 299 | 0.000 |
| 427 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т15: 485 → 0; т16: 0 → 485 | 0.001 |
| 428 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т32: 338 → 169 | 0.000 |
| 429 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т31: 0 → 169; т32: 169 → 0 | 0.003 |
| 430 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т33: 68 → 0; т34: 68 → 135 | 0.000 |
| 431 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т34: 135 → 68; т35: 68 → 135 | 0.000 |
| 432 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т35: 135 → 0; т37: 0 → 135 | 0.000 |
| 433 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т35: 0 → 68; т36: 68 → 0 | 0.001 |
| 434 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 0 → 68; т37: 68 → 0 | 0.001 |
| 435 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т37: 0 → 68; т38: 68 → 0 | 0.000 |
| 436 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т38: 0 → 68; т39: 68 → 0 | 0.000 |
| 437 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т40: 68 → 34 | 0.000 |
| 438 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т40: 34 → 68; т41: 68 → 34 | 0.000 |
| 439 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т41: 34 → 0; т42: 68 → 101 | 0.000 |
| 440 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т44: 68 → 0; т45: 68 → 135 | 0.000 |
| 441 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т45: 135 → 0; т46: 68 → 202 | 0.000 |
| 442 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т46: 202 → 0; т47: 0 → 202 | 0.000 |
| 443 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т46: 56 → 0 | 0.000 |
| 444 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т47: 60 → 0 | 0.000 |
| 445 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т47: 60 → 30 | 0.000 |
| 446 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 40 → 99; т8: 59 → 0 | 0.000 |
| 447 | x1.5 | valve | crude | `tg.tb.term_tw.grid_tw` | т14: 54 → 82 | 0.000 |
| 448 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т1: 54 → 0; т2: 0 → 54 | 0.000 |
| 449 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т39: 47 → 0 | 0.000 |
| 450 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т2: 387 → 193; т3: 0 → 193 | 0.000 |
| 451 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т2: 193 → 97; т3: 193 → 290 | 0.000 |
| 452 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т40: 38 → 0 | 0.000 |
| 453 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т10: 32 → 0; т11: 0 → 32 | 0.000 |
| 454 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 30 → 46 | 0.063 |
| 455 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 46 → 68 | 0.094 |
| 456 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т1: 68 → 103 | 0.058 |
| 457 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т2: 30 → 0; т3: 30 → 61 | 0.000 |
| 458 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т3: 61 → 0; т4: 410 → 471 | 0.000 |
| 459 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т41: 19 → 0 | 0.000 |
| 460 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т27: 19 → 0; т28: 0 → 19 | 0.000 |
| 461 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т44: 11 → 0 | 0.000 |
| 462 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т27: 252 → 0; т28: 0 → 252 | 0.000 |
| 463 | drop | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т43: 4 → 0 | 0.000 |
| 464 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т42: 3 → 0 | 0.000 |
| 465 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т50: 8,629 → 17,384; т51: 17,511 → 8,755 | 0.000 |
| 466 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т51: 8,755 → 13,133 | 0.000 |
| 467 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т11: 0 → 5,330; т12: 10,659 → 5,330 | 0.004 |
| 468 | half | valve | lng | `tg.tb.term_kr.grid_kr` | т12: 5,330 → 2,665 | 0.773 |
| 469 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т47: 8,629 → 12,943 | 0.000 |
| 470 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т38: 8,621 → 12,932 | 0.000 |
| 471 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т3: 8,134 → 12,201 | 0.000 |
| 472 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т2: 3,111 → 9,211; т3: 12,201 → 6,100 | 0.000 |
| 473 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т3: 6,100 → 3,050; т4: 0 → 3,050 | 0.000 |
| 474 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 8,100 → 12,150 | 0.000 |
| 475 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т19: 0 → 12,150; т20: 12,150 → 0 | 0.000 |
| 476 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 0 → 6,480; т35: 6,480 → 0 | 0.000 |
| 477 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т6: 5,400 → 0; т7: 0 → 5,400 | 0.000 |
| 478 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т25: 5,287 → 0; т26: 0 → 5,287 | 0.000 |
| 479 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т5: 4,108 → 2,054; т6: 0 → 2,054 | 0.000 |
| 480 | shift+2 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 4,100 → 0; т24: 0 → 4,100 | 0.000 |
| 481 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 0 → 1,982; т36: 3,963 → 1,982 | 0.000 |
| 482 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 0 → 1,870; т23: 3,740 → 1,870 | 0.000 |
| 483 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т18: 3,182 → 1,591; т19: 0 → 1,591 | 0.860 |
| 484 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т29: 3,182 → 1,591; т30: 0 → 1,591 | 0.893 |
| 485 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 3,095 → 1,548; т28: 0 → 1,548 | 0.000 |
| 486 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 1,548 → 774; т28: 1,548 → 2,321 | 0.000 |
| 487 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 774 → 387; т28: 2,321 → 2,708 | 0.000 |
| 488 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т6: 2,701 → 0; т7: 0 → 2,701 | 0.005 |
| 489 | half+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т24: 2,640 → 1,320; т25: 0 → 1,320 | 0.026 |
| 490 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 2,527 → 1,264; т38: 327 → 1,591 | 0.000 |
| 491 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 1,264 → 632 | 0.002 |
| 492 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т37: 632 → 0; т38: 1,591 → 2,223 | 0.000 |
| 493 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т42: 2,308 → 3,462 | 0.001 |
| 494 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т45: 2,287 → 3,431 | 0.169 |
| 495 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т6: 2,198 → 0; т7: 1,399 → 3,597 | 0.046 |
| 496 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т46: 2,110 → 3,165 | 0.006 |
| 497 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т41: 2,095 → 0; т42: 466 → 2,560 | 0.001 |
| 498 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т35: 1,971 → 2,957 | 0.104 |
| 499 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т5: 2,025 → 3,038 | 0.000 |
| 500 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т5: 3,038 → 1,519; т6: 0 → 1,519 | 0.000 |
| 501 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т38: 1,862 → 931; т39: 931 → 1,862 | 0.006 |
| 502 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т38: 931 → 466; т39: 1,862 → 2,328 | 0.009 |
| 503 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т29: 1,757 → 0; т30: 0 → 1,757 | 0.000 |
| 504 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т25: 1,624 → 0; т26: 0 → 1,624 | 0.000 |
| 505 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т10: 1,577 → 0; т11: 0 → 1,577 | 0.000 |
| 506 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т26: 1,576 → 788; т27: 387 → 1,175 | 0.054 |
| 507 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т26: 788 → 1,182 | 0.004 |
| 508 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т22: 1,504 → 752; т23: 804 → 1,556 | 0.000 |
| 509 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т5: 1,560 → 2,340 | 0.000 |
| 510 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т7: 3,597 → 1,798; т8: 0 → 1,798 | 0.114 |
| 511 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т7: 1,798 → 899; т8: 1,798 → 2,698 | 0.000 |
| 512 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т5: 1,371 → 2,270; т7: 899 → 0 | 0.060 |
| 513 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т43: 1,397 → 0; т44: 0 → 1,397 | 0.000 |
| 514 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т36: 1,360 → 0; т37: 0 → 1,360 | 0.006 |
| 515 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т16: 1,351 → 2,027 | 0.432 |
| 516 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т16: 2,027 → 1,013; т17: 930 → 1,944 | 0.000 |
| 517 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т41: 1,330 → 1,995 | 0.000 |
| 518 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т32: 1,326 → 663 | 0.040 |
| 519 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т32: 663 → 0; т33: 0 → 663 | 0.000 |
| 520 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 1,350 → 0; т24: 0 → 1,350 | 0.000 |
| 521 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т22: 0 → 675; т23: 1,350 → 675 | 0.000 |
| 522 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т11: 299 → 1,621; т12: 1,321 → 0 | 0.000 |
| 523 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т12: 0 → 661; т13: 1,321 → 661 | 0.000 |
| 524 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 1,197 → 599; т19: 0 → 599 | 0.000 |
| 525 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 599 → 299; т19: 599 → 898 | 0.000 |
| 526 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т18: 299 → 150 | 0.103 |
| 527 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т44: 1,167 → 1,751 | 0.000 |
| 528 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т43: 167 → 1,918; т44: 1,751 → 0 | 0.002 |
| 529 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т22: 1,248 → 1,872 | 0.000 |
| 530 | half+1 | valve | crude | `tg.tb.term_eu.grid_eu` | т37: 1,248 → 624; т38: 1,248 → 1,872 | 0.006 |
| 531 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т41: 1,248 → 0; т43: 0 → 1,248 | 0.000 |
| 532 | shift-1 | valve | crude | `tg.tb.term_eu.grid_eu` | т44: 1,248 → 2,496; т45: 1,248 → 0 | 0.000 |
| 533 | half+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т46: 1,248 → 624; т47: 676 → 1,300 | 0.000 |
| 534 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т46: 624 → 0; т48: 0 → 624 | 0.000 |
| 535 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 1,053 → 0; т10: 0 → 1,053 | 0.000 |
| 536 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 826 → 1,359; т30: 1,066 → 533 | 0.000 |
| 537 | shift-2 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 0 → 533; т30: 533 → 0 | 0.000 |
| 538 | shift-2 | valve | crude | `tg.tb.term_eu.grid_eu` | т27: 260 → 1,300; т29: 1,040 → 0 | 0.004 |
| 539 | x1.5 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 951 → 1,427 | 0.000 |
| 540 | half+1 | valve | lng | `tg.tb.term_tw.grid_tw` | т5: 1,427 → 714; т6: 0 → 714 | 0.000 |
| 541 | drop | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 1,012 → 0 | 0.811 |
| 542 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т39: 2,328 → 1,164; т40: 931 → 2,095 | 0.002 |
| 543 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т40: 2,095 → 1,047; т41: 0 → 1,047 | 0.003 |
| 544 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т17: 1,944 → 972; т18: 150 → 1,121 | 0.000 |
| 545 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т25: 900 → 0; т26: 0 → 900 | 0.000 |
| 546 | shift+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т38: 900 → 0; т39: 338 → 1,238 | 0.127 |
| 547 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т23: 1,556 → 778; т24: 787 → 1,564 | 0.000 |
| 548 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т28: 533 → 1,213; т29: 1,359 → 680 | 0.000 |
| 549 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 680 → 340; т30: 0 → 340 | 0.000 |
| 550 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т29: 340 → 510 | 0.000 |
| 551 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т36: 733 → 0; т37: 0 → 733 | 0.013 |
| 552 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т21: 678 → 339; т22: 752 → 1,091 | 0.010 |
| 553 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т43: 635 → 0; т44: 0 → 635 | 0.000 |
| 554 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 635 → 317 | 0.018 |
| 555 | shift+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т44: 676 → 0; т45: 0 → 676 | 0.000 |
| 556 | half+1 | order | crude | `sea.tb.src_us_crude.term_eu` | т47: 1,300 → 650; т48: 624 → 1,274 | 0.000 |
| 557 | shift+2 | order | crude | `sea.tb.src_us_crude.term_eu` | т47: 650 → 0; т49: 0 → 650 | 0.000 |
| 558 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т12: 262 → 597; т13: 669 → 335 | 0.326 |
| 559 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т13: 335 → 167; т14: 0 → 167 | 0.000 |
| 560 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т13: 167 → 251 | 0.019 |
| 561 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т42: 619 → 0; т43: 0 → 619 | 0.000 |
| 562 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т43: 554 → 830 | 0.049 |
| 563 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т41: 553 → 830 | 0.000 |
| 564 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т41: 830 → 1,245 | 0.000 |
| 565 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т40: 189 → 812; т41: 1,245 → 623 | 0.000 |
| 566 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т28: 551 → 0; т29: 0 → 551 | 0.000 |
| 567 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 528 → 792 | 0.608 |
| 568 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т13: 792 → 0; т14: 0 → 792 | 0.000 |
| 569 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т42: 2,560 → 1,280; т43: 0 → 1,280 | 0.000 |
| 570 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т1: 494 → 0; т2: 219 → 713 | 0.072 |
| 571 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т35: 110 → 598; т36: 488 → 0 | 0.029 |
| 572 | x2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т35: 486 → 972 | 0.079 |
| 573 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т35: 972 → 1,458 | 0.024 |
| 574 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т34: 410 → 1,868; т35: 1,458 → 0 | 0.000 |
| 575 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т36: 486 → 243 | 0.000 |
| 576 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т36: 243 → 122 | 0.000 |
| 577 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т16: 485 → 727 | 0.309 |
| 578 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т41: 450 → 0; т42: 0 → 450 | 0.000 |
| 579 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т15: 450 → 676 | 0.058 |
| 580 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т15: 676 → 338; т16: 1,013 → 1,351 | 0.000 |
| 581 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т15: 338 → 169 | 0.171 |
| 582 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т40: 446 → 0; т41: 0 → 446 | 0.000 |
| 583 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 471 → 235; т5: 410 → 645 | 0.000 |
| 584 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т4: 235 → 118; т5: 645 → 763 | 0.000 |
| 585 | shift+2 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 450 → 0; т38: 0 → 450 | 0.000 |
| 586 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т36: 0 → 225; т37: 450 → 225 | 0.000 |
| 587 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т2: 449 → 674 | 0.000 |
| 588 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т1: 0 → 674; т2: 674 → 0 | 0.000 |
| 589 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т23: 414 → 0; т24: 0 → 414 | 0.000 |
| 590 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т13: 411 → 205; т14: 1,014 → 1,219 | 0.002 |
| 591 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т30: 0 → 427; т31: 427 → 0 | 0.060 |
| 592 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 763 → 382; т6: 0 → 382 | 0.000 |
| 593 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т5: 382 → 191; т6: 382 → 572 | 0.000 |
| 594 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т34: 1,868 → 2,802 | 0.003 |
| 595 | half-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 289 → 1,690; т34: 2,802 → 1,401 | 0.000 |
| 596 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т34: 1,401 → 2,102 | 0.004 |
| 597 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т12: 406 → 203; т13: 0 → 203 | 0.000 |
| 598 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т12: 203 → 102 | 0.001 |
| 599 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т5: 405 → 608 | 0.084 |
| 600 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т4: 135 → 742; т5: 608 → 0 | 0.096 |
| 601 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т6: 403 → 0; т7: 51 → 454 | 0.000 |
| 602 | shift-2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т13: 251 → 653; т15: 402 → 0 | 0.004 |
| 603 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т25: 397 → 0; т26: 217 → 614 | 0.000 |
| 604 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т35: 376 → 188; т36: 193 → 381 | 0.011 |
| 605 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т22: 192 → 563; т23: 371 → 0 | 0.122 |
| 606 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т31: 338 → 0; т32: 0 → 338 | 0.000 |
| 607 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т33: 338 → 0; т34: 317 → 655 | 0.000 |
| 608 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т31: 335 → 0; т32: 0 → 335 | 0.000 |
| 609 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т28: 335 → 0; т29: 0 → 335 | 0.000 |
| 610 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т27: 357 → 0; т28: 174 → 531 | 0.015 |
| 611 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т25: 332 → 0; т26: 0 → 332 | 0.000 |
| 612 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т39: 327 → 0; т40: 0 → 327 | 0.000 |
| 613 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т19: 337 → 0; т20: 0 → 337 | 0.007 |
| 614 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т21: 312 → 0; т22: 0 → 312 | 0.000 |
| 615 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т20: 330 → 495 | 0.000 |
| 616 | half-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т19: 0 → 248; т20: 495 → 248 | 0.000 |
| 617 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т15: 321 → 482 | 0.464 |
| 618 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т14: 307 → 153; т15: 482 → 635 | 0.000 |
| 619 | shift-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т10: 0 → 1,621; т11: 1,621 → 0 | 0.000 |
| 620 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т3: 290 → 435 | 0.137 |
| 621 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т3: 435 → 218; т4: 201 → 418 | 0.000 |
| 622 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т44: 13 → 302; т45: 289 → 0 | 0.000 |
| 623 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 1,690 → 2,534 | 0.002 |
| 624 | x2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 2,534 → 5,069 | 0.010 |
| 625 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т6: 270 → 0; т7: 270 → 540 | 0.000 |
| 626 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т8: 270 → 0; т9: 135 → 405 | 0.000 |
| 627 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т16: 269 → 0; т17: 0 → 269 | 0.000 |
| 628 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т28: 252 → 0; т29: 0 → 252 | 0.000 |
| 629 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т1: 0 → 237; т2: 237 → 0 | 0.080 |
| 630 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 221 → 110; т23: 0 → 110 | 0.000 |
| 631 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 110 → 55; т23: 110 → 165 | 0.000 |
| 632 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т22: 55 → 28; т23: 165 → 193 | 0.000 |
| 633 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т13: 234 → 117; т14: 153 → 270 | 0.000 |
| 634 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т13: 117 → 58; т14: 270 → 329 | 0.000 |
| 635 | drop | valve | crude | `tg.tb.term_kr.grid_kr` | т34: 220 → 0 | 0.004 |
| 636 | shift+1 | valve | crude | `tg.tb.term_kr.grid_kr` | т12: 220 → 0; т13: 0 → 220 | 0.001 |
| 637 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т30: 0 → 214; т31: 214 → 0 | 0.000 |
| 638 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т47: 202 → 0 | 0.000 |
| 639 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т22: 202 → 0; т23: 0 → 202 | 0.000 |
| 640 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т4: 418 → 0; т5: 29 → 447 | 0.000 |
| 641 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т28: 531 → 728; т29: 197 → 0 | 0.001 |
| 642 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т1: 177 → 265 | 0.353 |
| 643 | x2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т1: 265 → 530 | 0.306 |
| 644 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т1: 530 → 265; т2: 177 → 441 | 0.000 |
| 645 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т2: 441 → 221; т3: 177 → 397 | 0.000 |
| 646 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т2: 221 → 331 | 0.282 |
| 647 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т2: 331 → 166; т3: 397 → 563 | 0.000 |
| 648 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т3: 563 → 281; т4: 177 → 458 | 0.000 |
| 649 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т3: 281 → 141; т4: 458 → 599 | 0.000 |
| 650 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т4: 599 → 299; т5: 177 → 476 | 0.000 |
| 651 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т4: 299 → 150; т5: 476 → 626 | 0.000 |
| 652 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т5: 626 → 313; т6: 177 → 489 | 0.000 |
| 653 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т6: 489 → 245; т7: 177 → 421 | 0.000 |
| 654 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т6: 245 → 122; т7: 421 → 544 | 0.000 |
| 655 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т6: 122 → 61; т7: 544 → 605 | 0.000 |
| 656 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т7: 605 → 302; т8: 177 → 479 | 0.000 |
| 657 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т7: 302 → 151; т8: 479 → 630 | 0.000 |
| 658 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т8: 630 → 315; т9: 177 → 492 | 0.000 |
| 659 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т9: 492 → 246; т10: 177 → 422 | 0.000 |
| 660 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т9: 246 → 123; т10: 422 → 545 | 0.000 |
| 661 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т9: 123 → 61; т10: 545 → 607 | 0.000 |
| 662 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т10: 607 → 303; т11: 177 → 480 | 0.000 |
| 663 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т10: 303 → 152; т11: 480 → 632 | 0.000 |
| 664 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т11: 632 → 316; т12: 177 → 492 | 0.000 |
| 665 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т12: 492 → 246; т13: 177 → 423 | 0.000 |
| 666 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т12: 246 → 123; т13: 423 → 546 | 0.000 |
| 667 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т12: 123 → 62; т13: 546 → 607 | 0.000 |
| 668 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т13: 607 → 304; т14: 177 → 480 | 0.000 |
| 669 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т13: 304 → 152; т14: 480 → 632 | 0.000 |
| 670 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т14: 632 → 316; т15: 177 → 493 | 0.000 |
| 671 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т15: 493 → 246; т16: 177 → 423 | 0.000 |
| 672 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т15: 246 → 123; т16: 423 → 546 | 0.000 |
| 673 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т15: 123 → 62; т16: 546 → 608 | 0.000 |
| 674 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т16: 608 → 304; т17: 177 → 480 | 0.000 |
| 675 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т16: 304 → 152; т17: 480 → 632 | 0.000 |
| 676 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т17: 632 → 316; т18: 177 → 493 | 0.000 |
| 677 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 493 → 246; т19: 177 → 423 | 0.000 |
| 678 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 246 → 123; т19: 423 → 546 | 0.000 |
| 679 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т18: 123 → 62; т19: 546 → 608 | 0.000 |
| 680 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т19: 608 → 304; т20: 177 → 480 | 0.000 |
| 681 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т19: 304 → 152; т20: 480 → 632 | 0.000 |
| 682 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т20: 632 → 316; т21: 177 → 493 | 0.000 |
| 683 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 493 → 246; т22: 177 → 423 | 0.000 |
| 684 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 246 → 123; т22: 423 → 546 | 0.000 |
| 685 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т21: 123 → 62; т22: 546 → 608 | 0.000 |
| 686 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т22: 608 → 304; т23: 177 → 480 | 0.000 |
| 687 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т22: 304 → 152; т23: 480 → 632 | 0.000 |
| 688 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т23: 632 → 316; т24: 177 → 493 | 0.000 |
| 689 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 493 → 246; т25: 177 → 423 | 0.000 |
| 690 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 246 → 123; т25: 423 → 546 | 0.000 |
| 691 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т24: 123 → 62; т25: 546 → 608 | 0.000 |
| 692 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т25: 608 → 304; т26: 177 → 480 | 0.000 |
| 693 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т25: 304 → 152; т26: 480 → 632 | 0.000 |
| 694 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т26: 632 → 316; т27: 177 → 493 | 0.000 |
| 695 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т27: 493 → 246; т28: 177 → 423 | 0.000 |
| 696 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т27: 246 → 123; т28: 423 → 546 | 0.000 |
| 697 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т27: 123 → 62; т28: 546 → 608 | 0.000 |
| 698 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т28: 608 → 304; т29: 177 → 480 | 0.000 |
| 699 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т28: 304 → 152; т29: 480 → 632 | 0.000 |
| 700 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т29: 632 → 316; т30: 177 → 493 | 0.000 |
| 701 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т32: 177 → 0; т33: 177 → 353 | 0.000 |
| 702 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т33: 353 → 0; т34: 177 → 530 | 0.000 |
| 703 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т35: 177 → 0; т36: 177 → 353 | 0.000 |
| 704 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т36: 353 → 0; т37: 177 → 530 | 0.002 |
| 705 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 530 → 265; т38: 177 → 441 | 0.000 |
| 706 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 265 → 132; т38: 441 → 574 | 0.000 |
| 707 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т37: 132 → 66; т38: 574 → 640 | 0.000 |
| 708 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т38: 185 → 278 | 0.070 |
| 709 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т38: 278 → 417 | 0.156 |
| 710 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т16: 184 → 0; т17: 92 → 277 | 0.231 |
| 711 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т2: 183 → 0; т3: 0 → 183 | 0.000 |
| 712 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu]` | т43: 1,918 → 959; т44: 0 → 959 | 0.000 |
| 713 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т39: 177 → 0 | 0.000 |
| 714 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т40: 177 → 0; т41: 177 → 353 | 0.000 |
| 715 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т41: 353 → 177; т42: 177 → 353 | 0.000 |
| 716 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т41: 177 → 88 | 0.000 |
| 717 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т42: 353 → 177; т43: 173 → 350 | 0.000 |
| 718 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т3: 176 → 0; т4: 0 → 176 | 0.000 |
| 719 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т43: 350 → 175; т44: 166 → 341 | 0.000 |
| 720 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т31: 170 → 0; т32: 0 → 170 | 0.000 |
| 721 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т31: 169 → 253 | 0.089 |
| 722 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т31: 253 → 0; т32: 0 → 253 | 0.000 |
| 723 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т44: 341 → 170; т45: 0 → 170 | 0.000 |
| 724 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т33: 150 → 75; т34: 0 → 75 | 0.001 |
| 725 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_eu.cape]` | т33: 75 → 38 | 0.012 |
| 726 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т32: 152 → 76 | 0.000 |
| 727 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т32: 76 → 38 | 0.000 |
| 728 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т3: 202 → 945; т4: 742 → 0 | 0.009 |
| 729 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т37: 135 → 0; т38: 0 → 135 | 0.000 |
| 730 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т8: 121 → 0; т9: 65 → 186 | 0.000 |
| 731 | shift-1 | valve | crude | `tg.tb.term_kr.grid_kr` | т32: 327 → 437; т33: 110 → 0 | 0.002 |
| 732 | x1.5 | valve | crude | `tg.tb.term_eu.grid_eu` | т48: 104 → 156 | 0.000 |
| 733 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т42: 101 → 0; т43: 68 → 169 | 0.000 |
| 734 | x1.5 | valve | crude | `tg.tb.term_kr.grid_kr` | т7: 99 → 148 | 0.019 |
| 735 | x1.5 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т2: 97 → 145 | 0.037 |
| 736 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т19: 91 → 0; т21: 0 → 91 | 0.000 |
| 737 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т34: 68 → 0 | 0.000 |
| 738 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т35: 68 → 0; т37: 0 → 68 | 0.000 |
| 739 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т36: 68 → 135; т37: 68 → 0 | 0.001 |
| 740 | shift-1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т37: 0 → 68; т38: 68 → 0 | 0.007 |
| 741 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т40: 68 → 0; т41: 0 → 68 | 0.000 |
| 742 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp]` | т43: 169 → 0; т44: 0 → 169 | 0.000 |
| 743 | shift+2 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr]` | т2: 54 → 0; т4: 176 → 231 | 0.000 |
| 744 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т7: 454 → 227; т8: 0 → 227 | 0.000 |
| 745 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т20: 36 → 0 | 0.031 |
| 746 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т11: 32 → 0; т12: 213 → 245 | 0.000 |
| 747 | drop | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.east]` | т47: 30 → 0 | 0.000 |
| 748 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_kr.lombok]` | т24: 26 → 0; т25: 0 → 26 | 0.000 |
| 749 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.lombok]` | т28: 19 → 0; т29: 0 → 19 | 0.000 |
| 750 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т31: 7 → 10 | 0.002 |
| 751 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т31: 10 → 15 | 0.003 |
| 752 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.east]` | т31: 15 → 23 | 0.005 |
| 753 | half-1 | valve | lng | `tg.tb.term_eu.grid_eu` | т49: 8,629 → 17,321; т50: 17,384 → 8,692 | 0.000 |
| 754 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т18: 0 → 12,150; т19: 12,150 → 0 | 0.000 |
| 755 | x1.5 | valve | lng | `tg.tb.term_eu.grid_eu` | т34: 8,756 → 13,134 | 0.000 |
| 756 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т33: 0 → 6,480; т34: 6,480 → 0 | 0.000 |
| 757 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т7: 5,400 → 0; т8: 0 → 5,400 | 0.000 |
| 758 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т11: 5,330 → 7,994 | 0.001 |
| 759 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т26: 5,287 → 0; т27: 0 → 5,287 | 0.000 |
| 760 | half | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т33: 5,069 → 2,534 | 0.022 |
| 761 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т24: 4,100 → 0; т25: 0 → 4,100 | 0.000 |
| 762 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т21: 2,734 → 4,100 | 0.000 |
| 763 | shift-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т20: 0 → 4,100; т21: 4,100 → 0 | 0.000 |
| 764 | shift+1 | valve | lng | `tg.tb.term_kr.grid_kr` | т7: 2,701 → 0; т8: 2,357 → 5,058 | 0.002 |
| 765 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т8: 2,698 → 1,349; т9: 0 → 1,349 | 0.000 |
| 766 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т5: 2,270 → 1,135; т6: 0 → 1,135 | 0.000 |
| 767 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т5: 1,135 → 568; т6: 1,135 → 1,703 | 0.000 |
| 768 | half+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т20: 2,278 → 1,139; т21: 91 → 1,230 | 0.016 |
| 769 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т6: 2,054 → 1,027; т7: 0 → 1,027 | 0.000 |
| 770 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т6: 1,027 → 1,540 | 0.000 |
| 771 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 1,982 → 2,972 | 0.000 |
| 772 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 0 → 1,486; т35: 2,972 → 1,486 | 0.000 |
| 773 | shift+1 | order | crude | `sea.tb.src_gulf_crude.chk_hormuz [lane.src_gulf_crude.term_jp.east]` | т34: 2,102 → 0; т35: 0 → 2,102 | 0.001 |
| 774 | shift-1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т34: 655 → 2,560; т35: 1,905 → 0 | 0.005 |
| 775 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т22: 1,870 → 2,805 | 0.000 |
| 776 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т21: 0 → 1,402; т22: 2,805 → 1,402 | 0.000 |
| 777 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 1,870 → 935; т24: 0 → 935 | 0.000 |
| 778 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т23: 935 → 1,402 | 0.000 |
| 779 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 1,624 → 812; т27: 0 → 812 | 0.000 |
| 780 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 812 → 406; т27: 812 → 1,218 | 0.000 |
| 781 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr]` | т26: 406 → 203; т27: 1,218 → 1,421 | 0.000 |
| 782 | x1.5 | valve | lng | `tg.tb.term_kr.grid_kr` | т30: 1,591 → 2,387 | 0.064 |
| 783 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т11: 1,577 → 788; т12: 0 → 788 | 0.000 |
| 784 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т9: 1,349 → 2,137; т11: 788 → 0 | 0.293 |
| 785 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 0 → 810; т10: 1,621 → 810 | 0.000 |
| 786 | half-1 | valve | crude | `tg.tb.term_jp.grid_jp` | т9: 810 → 1,215; т10: 810 → 405 | 0.000 |
| 787 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т6: 1,519 → 759; т7: 0 → 759 | 0.000 |
| 788 | x1.5 | valve | crude | `tg.tb.term_jp.grid_jp` | т6: 759 → 1,139 | 0.000 |
| 789 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т20: 1,411 → 705; т21: 339 → 1,044 | 0.000 |
| 790 | shift+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т44: 1,397 → 0; т45: 0 → 1,397 | 0.000 |
| 791 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т35: 1,360 → 680; т36: 0 → 680 | 0.000 |
| 792 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т37: 1,360 → 680; т38: 0 → 680 | 0.000 |
| 793 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т37: 680 → 340; т38: 680 → 1,020 | 0.000 |
| 794 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_tw]` | т37: 340 → 170; т38: 1,020 → 1,190 | 0.000 |
| 795 | half-1 | valve | lng | `tg.tb.term_kr.grid_kr` | т24: 1,320 → 1,980; т25: 1,320 → 660 | 0.000 |
| 796 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т42: 1,280 → 640; т43: 1,280 → 1,920 | 0.000 |
| 797 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т42: 640 → 320 | 0.001 |
| 798 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т43: 1,920 → 960; т44: 0 → 960 | 0.000 |
| 799 | half+1 | valve | crude | `tg.tb.term_jp.grid_jp` | т24: 1,350 → 675; т25: 0 → 675 | 0.000 |
| 800 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т14: 1,219 → 610; т15: 169 → 779 | 0.023 |
| 801 | x1.5 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 1,175 → 1,762 | 0.068 |
| 802 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 1,762 → 881; т28: 2,708 → 3,589 | 0.000 |
| 803 | half+1 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т27: 881 → 441; т28: 3,589 → 4,030 | 0.000 |
| 804 | half | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т39: 1,164 → 582 | 0.070 |
| 805 | shift-2 | order | lng | `sea.tb.src_qa_lng.chk_hormuz [lane.src_qa_lng.term_kr.lombok]` | т37: 0 → 582; т39: 582 → 0 | 0.019 |
