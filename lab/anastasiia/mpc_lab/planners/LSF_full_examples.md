# LSF: прийняті ходи паливного локального пошуку (planner_LSF.py)

Агент `anastasiia_hybrid_chiplp`, task full, root 444; 3 епізоди(ів) з найбільшим виграшем. Ходи в порядку прийняття; «тиждень» — тиждень відправки; кількість — у одиницях товару. Тип: valve — термінал → система (лаг 0), order — замовлення з джерела. Перевірено одним прогоном.

## Епізод 6: J 9039.45 → 9035.46 млрд USD (−3.99), прийнято 280 ходів

| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |
|---|---|---|---|---|---|---|
| 1 | half | order | nucfuel | `pipe.tb.src_kz_uranium.grid_us` | т1: 8,640 → 4,320 | 0.069 |
| 2 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_us` | т72: 8,640 → 0; т74: 1,561 → 10,201 | 0.009 |
| 3 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_us` | т73: 8,640 → 0; т75: 0 → 8,640 | 0.003 |
| 4 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 38,400 → 19,200 | 0.096 |
| 5 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 19,200 → 9,600 | 0.049 |
| 6 | shift-1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 9,600 → 48,000; т2: 38,400 → 0 | 0.048 |
| 7 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т5: 38,400 → 19,200 | 0.097 |
| 8 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т8: 38,400 → 19,200 | 0.097 |
| 9 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т11: 38,400 → 19,200 | 0.097 |
| 10 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т14: 38,400 → 19,200 | 0.097 |
| 11 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т17: 38,400 → 19,200 | 0.097 |
| 12 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т20: 38,400 → 19,200 | 0.097 |
| 13 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т23: 38,400 → 19,200 | 0.097 |
| 14 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т26: 38,400 → 19,200 | 0.097 |
| 15 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т29: 38,400 → 19,200 | 0.097 |
| 16 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т32: 38,400 → 19,200 | 0.097 |
| 17 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т35: 38,400 → 19,200 | 0.097 |
| 18 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т38: 38,400 → 19,200 | 0.097 |
| 19 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т41: 38,400 → 19,200 | 0.097 |
| 20 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т44: 38,400 → 19,200 | 0.097 |
| 21 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т47: 38,400 → 19,200 | 0.097 |
| 22 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т50: 38,400 → 19,200 | 0.097 |
| 23 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т53: 38,400 → 19,200 | 0.097 |
| 24 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т56: 38,400 → 19,200 | 0.097 |
| 25 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т59: 38,400 → 19,200 | 0.097 |
| 26 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т62: 38,400 → 19,200 | 0.097 |
| 27 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т65: 38,400 → 19,200 | 0.097 |
| 28 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т68: 38,400 → 19,200 | 0.097 |
| 29 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т71: 38,400 → 19,200 | 0.097 |
| 30 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т74: 38,400 → 19,200 | 0.097 |
| 31 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т77: 38,400 → 19,200 | 0.097 |
| 32 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т80: 38,400 → 19,200 | 0.096 |
| 33 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т83: 38,400 → 19,200 | 0.096 |
| 34 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т86: 38,400 → 19,200 | 0.096 |
| 35 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т89: 38,400 → 19,200 | 0.081 |
| 36 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т91: 38,400 → 19,200 | 0.008 |
| 37 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т93: 38,400 → 19,200 | 0.007 |
| 38 | half+1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 38,400 → 19,200; т101: 9,557 → 28,757 | 0.001 |
| 39 | half+1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 19,200 → 9,600; т101: 28,757 → 38,357 | 0.000 |
| 40 | shift+2 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 9,600 → 0; т102: 6,400 → 16,000 | 0.001 |
| 41 | half+1 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т73: 5,400 → 2,700; т74: 1,790 → 4,490 | 0.000 |
| 42 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т73: 2,700 → 0; т75: 0 → 2,700 | 0.001 |
| 43 | half | order | nucfuel | `pipe.tb.src_kz_uranium.grid_kr` | т1: 1,980 → 990 | 0.016 |
| 44 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_kr` | т36: 1,980 → 0; т38: 0 → 1,980 | 0.001 |
| 45 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 1,980 → 990; т37: 633 → 1,623 | 0.000 |
| 46 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 990 → 495; т37: 1,623 → 2,118 | 0.001 |
| 47 | shift+2 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 495 → 0; т38: 0 → 495 | 0.000 |
| 48 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_kr` | т37: 1,980 → 0; т39: 0 → 1,980 | 0.001 |
| 49 | half+1 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т74: 4,490 → 2,245; т75: 2,700 → 4,945 | 0.000 |
| 50 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т74: 2,245 → 0; т76: 0 → 2,245 | 0.001 |
| 51 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_us` | т74: 10,201 → 0; т76: 0 → 10,201 | 0.003 |
| 52 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т8: 7,020 → 10,530 | 0.005 |
| 53 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т8: 10,530 → 15,795 | 0.001 |
| 54 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т8: 15,795 → 7,898; т9: 3,780 → 11,677 | 0.001 |
| 55 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т10: 7,020 → 0; т11: 5,400 → 12,420 | 0.000 |
| 56 | half | order | lng | `pipe.tb.src_no_gas.grid_eu` | т14: 7,017 → 3,509 | 0.011 |
| 57 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т22: 7,017 → 3,509; т23: 7,017 → 10,526 | 0.000 |
| 58 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т23: 10,526 → 5,263; т24: 7,017 → 12,280 | 0.000 |
| 59 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т24: 12,280 → 6,140; т25: 7,017 → 13,157 | 0.000 |
| 60 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т25: 13,157 → 6,579; т26: 7,017 → 13,596 | 0.000 |
| 61 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т26: 13,596 → 6,798; т27: 7,017 → 13,815 | 0.000 |
| 62 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т27: 13,815 → 6,907; т28: 7,017 → 13,925 | 0.000 |
| 63 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т28: 13,925 → 6,962; т29: 7,017 → 13,979 | 0.000 |
| 64 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т28: 6,962 → 3,481; т29: 13,979 → 17,460 | 0.000 |
| 65 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т29: 17,460 → 8,730; т30: 7,017 → 15,747 | 0.000 |
| 66 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т30: 15,747 → 7,874; т31: 7,017 → 14,891 | 0.000 |
| 67 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т30: 7,874 → 3,937; т31: 14,891 → 18,828 | 0.000 |
| 68 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т31: 18,828 → 9,414; т32: 7,017 → 16,431 | 0.000 |
| 69 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т31: 9,414 → 4,707; т32: 16,431 → 21,138 | 0.004 |
| 70 | shift+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т33: 7,017 → 0; т34: 7,017 → 14,034 | 0.000 |
| 71 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т34: 14,034 → 7,017; т35: 7,017 → 14,034 | 0.000 |
| 72 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т34: 7,017 → 3,509; т35: 14,034 → 17,543 | 0.000 |
| 73 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т34: 3,509 → 1,754; т35: 17,543 → 19,297 | 0.000 |
| 74 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т35: 19,297 → 9,648; т36: 7,017 → 16,666 | 0.000 |
| 75 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т36: 16,666 → 8,333; т37: 7,017 → 15,350 | 0.000 |
| 76 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т36: 8,333 → 4,166; т37: 15,350 → 19,516 | 0.000 |
| 77 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т37: 19,516 → 9,758; т38: 7,017 → 16,775 | 0.000 |
| 78 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т38: 16,775 → 8,388; т39: 7,017 → 15,405 | 0.000 |
| 79 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т38: 8,388 → 4,194; т39: 15,405 → 19,598 | 0.000 |
| 80 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т39: 19,598 → 9,799; т40: 7,017 → 16,816 | 0.000 |
| 81 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т40: 16,816 → 8,408; т41: 7,017 → 15,425 | 0.000 |
| 82 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т40: 8,408 → 4,204; т41: 15,425 → 19,629 | 0.000 |
| 83 | shift+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т42: 7,017 → 0; т43: 7,017 → 14,034 | 0.000 |
| 84 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т43: 14,034 → 7,017; т44: 7,017 → 14,034 | 0.000 |
| 85 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т43: 7,017 → 3,509; т44: 14,034 → 17,543 | 0.000 |
| 86 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т43: 3,509 → 1,754; т44: 17,543 → 19,297 | 0.000 |
| 87 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т44: 19,297 → 9,648; т45: 7,017 → 16,666 | 0.000 |
| 88 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т45: 16,666 → 8,333; т46: 7,017 → 15,350 | 0.000 |
| 89 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т45: 8,333 → 4,166; т46: 15,350 → 19,516 | 0.000 |
| 90 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т46: 19,516 → 9,758; т47: 7,017 → 16,775 | 0.000 |
| 91 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т47: 16,775 → 8,388; т48: 7,017 → 15,405 | 0.000 |
| 92 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т47: 8,388 → 4,194; т48: 15,405 → 19,598 | 0.000 |
| 93 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т48: 19,598 → 9,799; т49: 7,017 → 16,816 | 0.000 |
| 94 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т49: 16,816 → 8,408; т50: 7,017 → 15,425 | 0.000 |
| 95 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т49: 8,408 → 4,204; т50: 15,425 → 19,629 | 0.000 |
| 96 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т50: 19,629 → 9,815; т51: 7,017 → 16,832 | 0.000 |
| 97 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т51: 16,832 → 8,416; т52: 7,017 → 15,433 | 0.000 |
| 98 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т51: 8,416 → 4,208; т52: 15,433 → 19,641 | 0.000 |
| 99 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т52: 19,641 → 9,820; т53: 7,017 → 16,838 | 0.000 |
| 100 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т53: 16,838 → 8,419; т54: 7,017 → 15,436 | 0.000 |
| 101 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т53: 8,419 → 4,209; т54: 15,436 → 19,645 | 0.000 |
| 102 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т54: 19,645 → 9,823; т55: 7,017 → 16,840 | 0.000 |
| 103 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т55: 16,840 → 8,420; т56: 7,017 → 15,437 | 0.000 |
| 104 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т55: 8,420 → 4,210; т56: 15,437 → 19,647 | 0.000 |
| 105 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т56: 19,647 → 9,823; т57: 7,017 → 16,840 | 0.000 |
| 106 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т57: 16,840 → 8,420; т58: 7,017 → 15,437 | 0.000 |
| 107 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т57: 8,420 → 4,210; т58: 15,437 → 19,647 | 0.000 |
| 108 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т58: 19,647 → 9,824; т59: 7,017 → 16,841 | 0.000 |
| 109 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т59: 16,841 → 8,420; т60: 7,017 → 15,437 | 0.000 |
| 110 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т59: 8,420 → 4,210; т60: 15,437 → 19,648 | 0.000 |
| 111 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т60: 19,648 → 9,824; т61: 7,017 → 16,841 | 0.000 |
| 112 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т61: 16,841 → 8,420; т62: 7,017 → 15,438 | 0.000 |
| 113 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т61: 8,420 → 4,210; т62: 15,438 → 19,648 | 0.000 |
| 114 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т62: 19,648 → 9,824; т63: 7,017 → 16,841 | 0.000 |
| 115 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т63: 16,841 → 8,420; т64: 7,017 → 15,438 | 0.000 |
| 116 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т63: 8,420 → 4,210; т64: 15,438 → 19,648 | 0.000 |
| 117 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т64: 19,648 → 9,824; т65: 7,017 → 16,841 | 0.000 |
| 118 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т65: 16,841 → 8,420; т66: 7,017 → 15,438 | 0.000 |
| 119 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т65: 8,420 → 4,210; т66: 15,438 → 19,648 | 0.000 |
| 120 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т66: 19,648 → 9,824; т67: 7,017 → 16,841 | 0.000 |
| 121 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т67: 16,841 → 8,420; т68: 7,017 → 15,438 | 0.000 |
| 122 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т67: 8,420 → 4,210; т68: 15,438 → 19,648 | 0.000 |
| 123 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т68: 19,648 → 9,824; т69: 7,017 → 16,841 | 0.000 |
| 124 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т69: 16,841 → 8,420; т70: 7,017 → 15,438 | 0.000 |
| 125 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т69: 8,420 → 4,210; т70: 15,438 → 19,648 | 0.000 |
| 126 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т70: 19,648 → 9,824; т71: 7,017 → 16,841 | 0.000 |
| 127 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т71: 16,841 → 8,420; т72: 7,017 → 15,438 | 0.000 |
| 128 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т71: 8,420 → 4,210; т72: 15,438 → 19,648 | 0.000 |
| 129 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т72: 19,648 → 9,824; т73: 7,017 → 16,841 | 0.000 |
| 130 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т73: 16,841 → 8,420; т74: 7,017 → 15,438 | 0.000 |
| 131 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т73: 8,420 → 4,210; т74: 15,438 → 19,648 | 0.000 |
| 132 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т74: 19,648 → 9,824; т75: 7,017 → 16,841 | 0.000 |
| 133 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т75: 16,841 → 8,420; т76: 7,017 → 15,438 | 0.000 |
| 134 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т75: 8,420 → 4,210; т76: 15,438 → 19,648 | 0.000 |
| 135 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т76: 19,648 → 9,824; т77: 7,017 → 16,841 | 0.000 |
| 136 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т77: 16,841 → 8,420; т78: 7,017 → 15,438 | 0.000 |
| 137 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т77: 8,420 → 4,210; т78: 15,438 → 19,648 | 0.000 |
| 138 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т78: 19,648 → 9,824; т79: 7,017 → 16,841 | 0.000 |
| 139 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т79: 16,841 → 8,420; т80: 7,017 → 15,438 | 0.000 |
| 140 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т79: 8,420 → 4,210; т80: 15,438 → 19,648 | 0.000 |
| 141 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т80: 19,648 → 9,824; т81: 7,017 → 16,841 | 0.000 |
| 142 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т81: 16,841 → 8,420; т82: 7,017 → 15,438 | 0.000 |
| 143 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т81: 8,420 → 4,210; т82: 15,438 → 19,648 | 0.000 |
| 144 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т82: 19,648 → 9,824; т83: 7,017 → 16,841 | 0.000 |
| 145 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т83: 16,841 → 8,420; т84: 7,017 → 15,438 | 0.000 |
| 146 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т83: 8,420 → 4,210; т84: 15,438 → 19,648 | 0.000 |
| 147 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т84: 19,648 → 9,824; т85: 7,017 → 16,841 | 0.000 |
| 148 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т85: 16,841 → 8,420; т86: 7,017 → 15,438 | 0.000 |
| 149 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т85: 8,420 → 4,210; т86: 15,438 → 19,648 | 0.000 |
| 150 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т86: 19,648 → 9,824; т87: 7,017 → 16,841 | 0.000 |
| 151 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т87: 16,841 → 8,420; т88: 7,017 → 15,438 | 0.000 |
| 152 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т87: 8,420 → 4,210; т88: 15,438 → 19,648 | 0.000 |
| 153 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т88: 19,648 → 9,824; т89: 7,017 → 16,841 | 0.000 |
| 154 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т89: 16,841 → 8,420; т90: 7,017 → 15,438 | 0.000 |
| 155 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т89: 8,420 → 4,210; т90: 15,438 → 19,648 | 0.000 |
| 156 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т90: 19,648 → 9,824; т91: 7,017 → 16,841 | 0.000 |
| 157 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т91: 16,841 → 8,420; т92: 7,017 → 15,438 | 0.000 |
| 158 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т91: 8,420 → 4,210; т92: 15,438 → 19,648 | 0.000 |
| 159 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т92: 19,648 → 9,824; т93: 7,017 → 16,841 | 0.000 |
| 160 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т93: 16,841 → 8,420; т94: 7,017 → 15,438 | 0.000 |
| 161 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т93: 8,420 → 4,210; т94: 15,438 → 19,648 | 0.000 |
| 162 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т94: 19,648 → 9,824; т95: 7,017 → 16,841 | 0.000 |
| 163 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т95: 16,841 → 8,420; т96: 7,017 → 15,438 | 0.000 |
| 164 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т95: 8,420 → 4,210; т96: 15,438 → 19,648 | 0.000 |
| 165 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т96: 19,648 → 9,824; т97: 7,017 → 16,841 | 0.000 |
| 166 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т97: 16,841 → 8,420; т98: 7,017 → 15,438 | 0.000 |
| 167 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т97: 8,420 → 4,210; т98: 15,438 → 19,648 | 0.000 |
| 168 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т98: 19,648 → 9,824; т99: 7,017 → 16,841 | 0.000 |
| 169 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т99: 16,841 → 8,420; т100: 2,419 → 10,840 | 0.000 |
| 170 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т99: 8,420 → 4,210; т100: 10,840 → 15,050 | 0.000 |
| 171 | half+1 | order | lng | `pipe.tb.src_no_gas.grid_eu` | т99: 4,210 → 2,105; т100: 15,050 → 17,155 | 0.000 |
| 172 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т16: 6,458 → 9,687 | 0.002 |
| 173 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т16: 9,687 → 14,531 | 0.001 |
| 174 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т15: 6,268 → 13,533; т16: 14,531 → 7,266 | 0.000 |
| 175 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т25: 6,358 → 9,537 | 0.000 |
| 176 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т25: 9,537 → 0; т26: 5,400 → 14,937 | 0.000 |
| 177 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т26: 14,937 → 0; т27: 5,400 → 20,337 | 0.000 |
| 178 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т27: 20,337 → 10,168; т28: 5,400 → 15,568 | 0.000 |
| 179 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т27: 10,168 → 15,253 | 0.000 |
| 180 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т27: 15,253 → 22,879 | 0.000 |
| 181 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т28: 15,568 → 0; т29: 5,400 → 20,968 | 0.000 |
| 182 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т29: 20,968 → 0; т30: 5,400 → 26,368 | 0.000 |
| 183 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т29: 0 → 13,184; т30: 26,368 → 13,184 | 0.000 |
| 184 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т30: 13,184 → 0; т31: 5,400 → 18,584 | 0.000 |
| 185 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т31: 18,584 → 0; т32: 5,400 → 23,984 | 0.000 |
| 186 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т31: 0 → 11,992; т32: 23,984 → 11,992 | 0.000 |
| 187 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т32: 11,992 → 0; т33: 5,400 → 17,392 | 0.000 |
| 188 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т33: 17,392 → 0; т34: 5,400 → 22,792 | 0.000 |
| 189 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т33: 0 → 11,396; т34: 22,792 → 11,396 | 0.000 |
| 190 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т34: 11,396 → 0; т35: 5,400 → 16,796 | 0.000 |
| 191 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 16,796 → 0; т36: 5,400 → 22,196 | 0.000 |
| 192 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т35: 0 → 11,098; т36: 22,196 → 11,098 | 0.000 |
| 193 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т36: 11,098 → 5,549; т37: 5,400 → 10,949 | 0.000 |
| 194 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т36: 5,549 → 8,324 | 0.000 |
| 195 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т37: 10,949 → 0; т38: 5,400 → 16,349 | 0.000 |
| 196 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т38: 16,349 → 0; т39: 5,400 → 21,749 | 0.000 |
| 197 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т40: 5,400 → 8,100 | 0.001 |
| 198 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т40: 8,100 → 0; т41: 5,400 → 13,500 | 0.000 |
| 199 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т41: 13,500 → 0; т42: 5,400 → 18,900 | 0.000 |
| 200 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т42: 18,900 → 28,350 | 0.000 |
| 201 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т41: 0 → 14,175; т42: 28,350 → 14,175 | 0.000 |
| 202 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т42: 14,175 → 0; т43: 5,400 → 19,575 | 0.000 |
| 203 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т43: 19,575 → 0; т44: 5,400 → 24,975 | 0.000 |
| 204 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т43: 0 → 12,488; т44: 24,975 → 12,488 | 0.000 |
| 205 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т44: 12,488 → 0; т45: 5,400 → 17,888 | 0.000 |
| 206 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т45: 17,888 → 0; т46: 5,400 → 23,288 | 0.000 |
| 207 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т45: 0 → 11,644; т46: 23,288 → 11,644 | 0.000 |
| 208 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т46: 11,644 → 0; т47: 5,400 → 17,044 | 0.000 |
| 209 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т47: 17,044 → 0; т48: 5,400 → 22,444 | 0.000 |
| 210 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т47: 0 → 11,222; т48: 22,444 → 11,222 | 0.000 |
| 211 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т48: 11,222 → 0; т49: 5,400 → 16,622 | 0.000 |
| 212 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т49: 16,622 → 0; т50: 5,400 → 22,022 | 0.000 |
| 213 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т50: 22,022 → 11,011; т51: 5,400 → 16,411 | 0.000 |
| 214 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т50: 11,011 → 16,516 | 0.000 |
| 215 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т50: 16,516 → 24,775 | 0.000 |
| 216 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т51: 16,411 → 0; т52: 5,400 → 21,811 | 0.000 |
| 217 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т52: 21,811 → 0; т53: 5,400 → 27,211 | 0.000 |
| 218 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т52: 0 → 13,605; т53: 27,211 → 13,605 | 0.000 |
| 219 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т53: 13,605 → 0; т54: 5,400 → 19,005 | 0.000 |
| 220 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т54: 19,005 → 0; т55: 5,400 → 24,405 | 0.000 |
| 221 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т54: 0 → 12,203; т55: 24,405 → 12,203 | 0.000 |
| 222 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т55: 12,203 → 0; т56: 5,400 → 17,603 | 0.000 |
| 223 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т56: 17,603 → 0; т57: 5,400 → 23,003 | 0.000 |
| 224 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т56: 0 → 11,501; т57: 23,003 → 11,501 | 0.000 |
| 225 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т57: 11,501 → 0; т58: 5,400 → 16,901 | 0.000 |
| 226 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т58: 16,901 → 0; т59: 5,400 → 22,301 | 0.000 |
| 227 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т58: 0 → 11,151; т59: 22,301 → 11,151 | 0.000 |
| 228 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т59: 11,151 → 0; т60: 5,400 → 16,551 | 0.000 |
| 229 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т60: 16,551 → 24,826 | 0.000 |
| 230 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т60: 24,826 → 0; т61: 5,400 → 30,226 | 0.000 |
| 231 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т60: 0 → 15,113; т61: 30,226 → 15,113 | 0.000 |
| 232 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т61: 15,113 → 0; т62: 5,400 → 20,513 | 0.000 |
| 233 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т62: 20,513 → 0; т63: 5,400 → 25,913 | 0.000 |
| 234 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т62: 0 → 12,957; т63: 25,913 → 12,957 | 0.000 |
| 235 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т63: 12,957 → 0; т64: 5,400 → 18,357 | 0.000 |
| 236 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т64: 18,357 → 0; т65: 5,400 → 23,757 | 0.000 |
| 237 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т64: 0 → 11,878; т65: 23,757 → 11,878 | 0.000 |
| 238 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т65: 11,878 → 0; т66: 5,400 → 17,278 | 0.000 |
| 239 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т66: 17,278 → 0; т67: 5,400 → 22,678 | 0.000 |
| 240 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т66: 0 → 11,339; т67: 22,678 → 11,339 | 0.000 |
| 241 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т67: 11,339 → 0; т68: 5,400 → 16,739 | 0.000 |
| 242 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т68: 16,739 → 25,109 | 0.000 |
| 243 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т68: 25,109 → 0; т69: 5,400 → 30,509 | 0.000 |
| 244 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т68: 0 → 15,254; т69: 30,509 → 15,254 | 0.000 |
| 245 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т69: 15,254 → 0; т70: 5,400 → 20,654 | 0.000 |
| 246 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т70: 20,654 → 0; т71: 5,400 → 26,054 | 0.000 |
| 247 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т70: 0 → 13,027; т71: 26,054 → 13,027 | 0.000 |
| 248 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т71: 13,027 → 0; т72: 5,400 → 18,427 | 0.000 |
| 249 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т72: 18,427 → 0; т73: 5,400 → 23,827 | 0.000 |
| 250 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т72: 0 → 11,914; т73: 23,827 → 11,914 | 0.000 |
| 251 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т73: 11,914 → 0; т74: 5,400 → 17,314 | 0.000 |
| 252 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т74: 17,314 → 0; т75: 5,400 → 22,714 | 0.000 |
| 253 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т75: 22,714 → 11,357; т76: 5,400 → 16,757 | 0.000 |
| 254 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т75: 11,357 → 17,035 | 0.000 |
| 255 | x1.5 | valve | lng | `tg.tb.term_jp.grid_jp` | т75: 17,035 → 25,553 | 0.000 |
| 256 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т76: 16,757 → 0; т77: 5,400 → 22,157 | 0.000 |
| 257 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т77: 22,157 → 0; т78: 5,400 → 27,557 | 0.000 |
| 258 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т77: 0 → 13,778; т78: 27,557 → 13,778 | 0.000 |
| 259 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т78: 13,778 → 0; т79: 5,400 → 19,178 | 0.000 |
| 260 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т79: 19,178 → 0; т80: 5,400 → 24,578 | 0.000 |
| 261 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т79: 0 → 12,289; т80: 24,578 → 12,289 | 0.000 |
| 262 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т80: 12,289 → 0; т81: 5,400 → 17,689 | 0.000 |
| 263 | shift+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т81: 17,689 → 0; т82: 5,400 → 23,089 | 0.000 |
| 264 | half-1 | valve | lng | `tg.tb.term_jp.grid_jp` | т81: 0 → 11,545; т82: 23,089 → 11,545 | 0.000 |
| 265 | half+1 | valve | lng | `tg.tb.term_jp.grid_jp` | т94: 5,346 → 2,673; т95: 5,346 → 8,018 | 0.000 |
| 266 | x1.5 | valve | lng | `tg.tb.term_cn.grid_cn` | т3: 5,068 → 7,602 | 0.000 |
| 267 | drop | valve | lng | `tg.tb.term_eu.grid_eu` | т104: 4,743 → 0 | 0.002 |
| 268 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т1: 4,587 → 2,293 | 0.013 |
| 269 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т1: 2,293 → 1,147 | 0.006 |
| 270 | drop | order | lng | `sea.tb.src_au_lng.term_jp` | т7: 4,587 → 0 | 0.025 |
| 271 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т8: 4,587 → 2,293 | 0.013 |
| 272 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т8: 2,293 → 1,147 | 0.097 |
| 273 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т10: 4,587 → 2,293 | 0.296 |
| 274 | drop | order | lng | `sea.tb.src_au_lng.term_jp` | т22: 4,587 → 0 | 0.025 |
| 275 | drop | order | lng | `sea.tb.src_au_lng.term_jp` | т23: 4,587 → 0 | 0.025 |
| 276 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т27: 4,587 → 2,293 | 0.013 |
| 277 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т29: 4,587 → 2,293 | 0.245 |
| 278 | drop | order | lng | `sea.tb.src_au_lng.term_jp` | т34: 4,587 → 0 | 0.025 |
| 279 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т35: 4,587 → 2,293 | 0.013 |
| 280 | half | order | lng | `sea.tb.src_au_lng.term_jp` | т35: 2,293 → 1,147 | 0.006 |

## Епізод 7: J 3866.32 → 3862.74 млрд USD (−3.58), прийнято 93 ходів

| # | хід | тип | паливо | слот | тиждень: було → стало | виграш, млрд |
|---|---|---|---|---|---|---|
| 1 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_us` | т49: 8,640 → 4,320; т50: 4,615 → 8,935 | 0.003 |
| 2 | shift+2 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_us` | т49: 4,320 → 0; т51: 0 → 4,320 | 0.001 |
| 3 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 38,400 → 19,200 | 0.096 |
| 4 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 19,200 → 9,600 | 0.049 |
| 5 | shift-1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т1: 9,600 → 48,000; т2: 38,400 → 0 | 0.048 |
| 6 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т5: 38,400 → 19,200 | 0.097 |
| 7 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т8: 38,400 → 19,200 | 0.096 |
| 8 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т11: 38,400 → 19,200 | 0.095 |
| 9 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т14: 38,400 → 19,200 | 0.094 |
| 10 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т17: 38,400 → 19,200 | 0.092 |
| 11 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т20: 38,400 → 19,200 | 0.091 |
| 12 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т20: 19,200 → 9,600 | 0.046 |
| 13 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т20: 9,600 → 4,800 | 0.023 |
| 14 | shift-1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т20: 4,800 → 43,200; т21: 38,400 → 0 | 0.022 |
| 15 | drop | order | lng | `pipe.tb.src_us_lng.grid_us` | т22: 38,400 → 0 | 0.187 |
| 16 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т23: 38,400 → 19,200 | 0.096 |
| 17 | shift-1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т23: 19,200 → 57,600; т24: 38,400 → 0 | 0.096 |
| 18 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т27: 38,400 → 19,200 | 0.097 |
| 19 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т30: 38,400 → 19,200 | 0.097 |
| 20 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т33: 38,400 → 19,200 | 0.097 |
| 21 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т36: 38,400 → 19,200 | 0.097 |
| 22 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т39: 38,400 → 19,200 | 0.097 |
| 23 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т42: 38,400 → 19,200 | 0.097 |
| 24 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т45: 38,400 → 19,200 | 0.097 |
| 25 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т48: 38,400 → 19,200 | 0.097 |
| 26 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т51: 38,400 → 19,200 | 0.097 |
| 27 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т54: 38,400 → 19,200 | 0.097 |
| 28 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т57: 38,400 → 19,200 | 0.097 |
| 29 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т60: 38,400 → 19,200 | 0.097 |
| 30 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т63: 38,400 → 19,200 | 0.097 |
| 31 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т66: 38,400 → 19,200 | 0.097 |
| 32 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т69: 38,400 → 19,200 | 0.097 |
| 33 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т72: 38,400 → 19,200 | 0.097 |
| 34 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т75: 38,400 → 19,200 | 0.097 |
| 35 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т78: 38,400 → 19,200 | 0.097 |
| 36 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т81: 38,400 → 19,200 | 0.097 |
| 37 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т84: 38,400 → 19,200 | 0.097 |
| 38 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т87: 38,400 → 19,200 | 0.097 |
| 39 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т90: 38,400 → 19,200 | 0.009 |
| 40 | half | order | lng | `pipe.tb.src_us_lng.grid_us` | т93: 38,400 → 19,200 | 0.007 |
| 41 | half+1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 38,400 → 19,200; т101: 9,557 → 28,757 | 0.001 |
| 42 | half+1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 19,200 → 9,600; т101: 28,757 → 38,357 | 0.000 |
| 43 | shift+2 | order | lng | `pipe.tb.src_us_lng.grid_us` | т100: 9,600 → 0; т102: 6,400 → 16,000 | 0.001 |
| 44 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_eu` | т73: 7,176 → 3,588; т74: 2,284 → 5,872 | 0.001 |
| 45 | half | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_eu` | т73: 3,588 → 1,794 | 0.007 |
| 46 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_eu` | т73: 1,794 → 897; т74: 5,872 → 6,769 | 0.000 |
| 47 | half+1 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т69: 5,400 → 2,700; т70: 109 → 2,809 | 0.000 |
| 48 | half+1 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т69: 2,700 → 1,350; т70: 2,809 → 4,159 | 0.000 |
| 49 | half+1 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_cn` | т69: 1,350 → 675; т70: 4,159 → 4,834 | 0.000 |
| 50 | shift+2 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_us` | т50: 8,935 → 0; т52: 0 → 8,935 | 0.003 |
| 51 | shift+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_eu` | т74: 6,769 → 0; т75: 0 → 6,769 | 0.001 |
| 52 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_kr` | т36: 1,980 → 0; т38: 0 → 1,980 | 0.001 |
| 53 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 1,980 → 990; т37: 299 → 1,289 | 0.000 |
| 54 | half+1 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 990 → 495; т37: 1,289 → 1,784 | 0.000 |
| 55 | shift+2 | order | nucfuel | `pipe.tb.src_ru_enrichment.grid_kr` | т36: 495 → 0; т38: 0 → 495 | 0.000 |
| 56 | shift+2 | order | nucfuel | `pipe.tb.src_kz_uranium.grid_kr` | т37: 1,980 → 0; т39: 0 → 1,980 | 0.001 |
| 57 | x1.5 | valve | lng | `tg.tb.term_sea.grid_sea` | т7: 9,764 → 14,646 | 0.006 |
| 58 | shift-1 | valve | lng | `tg.tb.term_sea.grid_sea` | т6: 5,145 → 19,791; т7: 14,646 → 0 | 0.008 |
| 59 | shift-1 | valve | lng | `tg.tb.term_sea.grid_sea` | т4: 7,426 → 17,189; т5: 9,764 → 0 | 0.009 |
| 60 | half+1 | order | lng | `pipe.tb.src_us_lng.grid_us` | т101: 38,357 → 19,179; т102: 16,000 → 35,179 | 0.001 |
| 61 | drop | order | lng | `sea.tb.src_au_lng.term_sea` | т5: 9,000 → 0 | 0.049 |
| 62 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т7: 9,000 → 4,500 | 0.021 |
| 63 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т7: 4,500 → 9,000; т8: 9,000 → 4,500 | 0.003 |
| 64 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т8: 4,500 → 2,250 | 0.007 |
| 65 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т8: 2,250 → 6,750; т9: 9,000 → 4,500 | 0.003 |
| 66 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т9: 4,500 → 9,000; т10: 9,000 → 4,500 | 0.001 |
| 67 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т22: 9,000 → 4,500 | 0.020 |
| 68 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т22: 4,500 → 9,000; т23: 9,000 → 4,500 | 0.003 |
| 69 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т23: 4,500 → 2,250 | 0.007 |
| 70 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т23: 2,250 → 6,750; т24: 9,000 → 4,500 | 0.004 |
| 71 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т24: 4,500 → 9,000; т25: 9,000 → 4,500 | 0.001 |
| 72 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т25: 4,500 → 2,250 | 0.009 |
| 73 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т25: 2,250 → 11,250; т26: 9,000 → 0 | 0.009 |
| 74 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т26: 0 → 9,000; т27: 9,000 → 0 | 0.004 |
| 75 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т27: 0 → 9,000; т28: 9,000 → 0 | 0.001 |
| 76 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т29: 9,000 → 4,500 | 0.018 |
| 77 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т29: 4,500 → 9,000; т30: 9,000 → 4,500 | 0.004 |
| 78 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т30: 4,500 → 9,000; т31: 9,000 → 4,500 | 0.001 |
| 79 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т31: 4,500 → 2,250 | 0.009 |
| 80 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т31: 2,250 → 11,250; т32: 9,000 → 0 | 0.009 |
| 81 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т32: 0 → 9,000; т33: 9,000 → 0 | 0.004 |
| 82 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т33: 0 → 9,000; т34: 9,000 → 0 | 0.001 |
| 83 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т35: 9,000 → 4,500 | 0.018 |
| 84 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т35: 4,500 → 9,000; т36: 9,000 → 4,500 | 0.004 |
| 85 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т36: 4,500 → 9,000; т37: 9,000 → 4,500 | 0.001 |
| 86 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т37: 4,500 → 2,250 | 0.008 |
| 87 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т37: 2,250 → 11,250; т38: 9,000 → 0 | 0.009 |
| 88 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т38: 0 → 9,000; т39: 9,000 → 0 | 0.004 |
| 89 | shift-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т39: 0 → 9,000; т40: 9,000 → 0 | 0.001 |
| 90 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т41: 9,000 → 4,500 | 0.017 |
| 91 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т41: 4,500 → 9,000; т42: 9,000 → 4,500 | 0.004 |
| 92 | half-1 | order | lng | `sea.tb.src_au_lng.term_sea` | т42: 4,500 → 9,000; т43: 9,000 → 4,500 | 0.001 |
| 93 | half | order | lng | `sea.tb.src_au_lng.term_sea` | т43: 4,500 → 2,250 | 0.008 |
