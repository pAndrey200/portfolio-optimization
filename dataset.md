https://drive.google.com/drive/folders/1Uw3SFii9-SY6ooxjozWwKulsn6vqD5o2?usp=sharing

Датасет main включает данные по акциям, облигациям и товарам (commodities). Имеется 322 156 строк и 26 столбцов. 

Основные признаки: 

Датасет:

* Ticker - биржевой символ
* Type - категория актива
* Date - дата, соответствующая каждому значению в рядах цен; данные за 10 лет с ежедневной частотой
* Open - цена открытия актива; имеются аномальные значения
* High - максимальная цена актива за день
* Low - минимальная цена актива за день; имеются аномальные значения
* Close - цена закрытия актива
* Volume - общее количество единиц актива, проданных за день
* Dividends - выплаченные дивиденды на акцию
* Stock Splits - деление акций; имеются записи до 20
* marketCap - рыночная капитализация
* forwardPE и trailingPE - прогнозный и исторический коэффициенты P/E
* dividendYield - доходность от дивидендов относительно цены

И другие: payoutRatio, priceToBook, bookValue (балансовая стоимость на акцию), debtToEquity, revenue (выручка), grossMargins, operatingMargins, profitMargins (маржинальные коэффициенты на различных этапах доходности), returnOnAssets (ROA), returnOnEquity (ROE), currentRatio и quickRatio (ликвидность).

Пропущенные значения присутствуют в столбцах, связанных с финансовыми показателями (marketCap, forwardPE, trailingPE, dividendYield и т. д.).

Подробнее: колонка, количесвто ненулевых значений, тип.

 0   Ticker      =      322156 non-null  object                          
 1   Type        =      322156 non-null  object                          
 2   Date        =      322156 non-null  datetime64[ns, America/New_York]
 3   Open       =      322156 non-null  float64                         
 4   High        =      322156 non-null  float64                         
 5   Low         =      322156 non-null  float64                         
 6   Close      =       322156 non-null  float64                         
 7   Volume    =       322156 non-null  int64                           
 8   Dividends  =     322156 non-null  float64                         
 9   Stock Splits  =  322156 non-null  float64                         
 10  marketCap   =  251700 non-null  float64                         
 11  forwardPE   =   261768 non-null  float64                         
 12  trailingPE    =   256734 non-null  float64                         
 13  dividendYield  =  239115 non-null  float64                         
 14  payoutRatio    =   239115 non-null  float64                         
 15  priceToBook   =    239115 non-null  float64                         
 16  bookValue      =    251700 non-null  float64                         
 17  debtToEquity  =    221496 non-null  float64                         
 18  revenue     =      251700 non-null  float64                         
 19  grossMargins   =   236598 non-null  float64                         
 20  operatingMargins = 251700 non-null  float64                         
 21  profitMargins   =  251700 non-null  float64                         
 22  returnOnAssets = 234081 non-null  float64                         
 23  returnOnEquity  =  224013 non-null  float64                         
 24  currentRatio   =   218979 non-null  float64                         
 25  quickRatio     =   218979 non-null  float64 

Детализация: 

stocks

1. AAPL - https://finance.yahoo.com/quote/AAPL
2. MSFT - https://finance.yahoo.com/quote/MSFT
3. GOOGL - https://finance.yahoo.com/quote/GOOGL
4. META - https://finance.yahoo.com/quote/META
5. NVDA - https://finance.yahoo.com/quote/NVDA
6. ORCL - https://finance.yahoo.com/quote/ORCL
7. IBM - https://finance.yahoo.com/quote/IBM
8. CSCO - https://finance.yahoo.com/quote/CSCO
9. INTC - https://finance.yahoo.com/quote/INTC
10. AMD - https://finance.yahoo.com/quote/AMD
11. JPM - https://finance.yahoo.com/quote/JPM
12. BAC - https://finance.yahoo.com/quote/BAC
13. C - https://finance.yahoo.com/quote/C
14. WFC - https://finance.yahoo.com/quote/WFC
15. GS - https://finance.yahoo.com/quote/GS
16. MS - https://finance.yahoo.com/quote/MS
17. AXP - https://finance.yahoo.com/quote/AXP
18. USB - https://finance.yahoo.com/quote/USB
19. PNC - https://finance.yahoo.com/quote/PNC
20. SCHW - https://finance.yahoo.com/quote/SCHW
21. XOM - https://finance.yahoo.com/quote/XOM
22. CVX - https://finance.yahoo.com/quote/CVX
23. COP - https://finance.yahoo.com/quote/COP
24. SLB - https://finance.yahoo.com/quote/SLB
25. HAL - https://finance.yahoo.com/quote/HAL
26. OXY - https://finance.yahoo.com/quote/OXY
27. EOG - https://finance.yahoo.com/quote/EOG
28. PSX - https://finance.yahoo.com/quote/PSX
29. VLO - https://finance.yahoo.com/quote/VLO
30. MPC - https://finance.yahoo.com/quote/MPC
31. AMZN - https://finance.yahoo.com/quote/AMZN
32. TSLA - https://finance.yahoo.com/quote/TSLA
33. HD - https://finance.yahoo.com/quote/HD
34. MCD - https://finance.yahoo.com/quote/MCD
35. NKE - https://finance.yahoo.com/quote/NKE
36. SBUX - https://finance.yahoo.com/quote/SBUX
37. TGT - https://finance.yahoo.com/quote/TGT
38. LOW - https://finance.yahoo.com/quote/LOW
39. GM - https://finance.yahoo.com/quote/GM
40. F - https://finance.yahoo.com/quote/F
41. PG - https://finance.yahoo.com/quote/PG
42. KO - https://finance.yahoo.com/quote/KO
43. PEP - https://finance.yahoo.com/quote/PEP
44. WMT - https://finance.yahoo.com/quote/WMT
45. COST - https://finance.yahoo.com/quote/COST
46. CL - https://finance.yahoo.com/quote/CL
47. KMB - https://finance.yahoo.com/quote/KMB
48. GIS - https://finance.yahoo.com/quote/GIS
49. MDLZ - https://finance.yahoo.com/quote/MDLZ
50. HSY - https://finance.yahoo.com/quote/HSY
51. JNJ - https://finance.yahoo.com/quote/JNJ
52. PFE - https://finance.yahoo.com/quote/PFE
53. MRK - https://finance.yahoo.com/quote/MRK
54. ABBV - https://finance.yahoo.com/quote/ABBV
55. BMY - https://finance.yahoo.com/quote/BMY
56. AMGN - https://finance.yahoo.com/quote/AMGN
57. GILD - https://finance.yahoo.com/quote/GILD
58. LLY - https://finance.yahoo.com/quote/LLY
59. MDT - https://finance.yahoo.com/quote/MDT
60. UNH - https://finance.yahoo.com/quote/UNH
61. BA - https://finance.yahoo.com/quote/BA
62. CAT - https://finance.yahoo.com/quote/CAT
63. GE - https://finance.yahoo.com/quote/GE
64. MMM - https://finance.yahoo.com/quote/MMM
65. HON - https://finance.yahoo.com/quote/HON
66. LMT - https://finance.yahoo.com/quote/LMT
67. DE - https://finance.yahoo.com/quote/DE
68. UPS - https://finance.yahoo.com/quote/UPS
69. FDX - https://finance.yahoo.com/quote/FDX
70. NOC - https://finance.yahoo.com/quote/NOC
71. NEE - https://finance.yahoo.com/quote/NEE
72. DUK - https://finance.yahoo.com/quote/DUK
73. SO - https://finance.yahoo.com/quote/SO
74. AEP - https://finance.yahoo.com/quote/AEP
75. EXC - https://finance.yahoo.com/quote/EXC
76. SRE - https://finance.yahoo.com/quote/SRE
77. PEG - https://finance.yahoo.com/quote/PEG
78. ED - https://finance.yahoo.com/quote/ED
79. XEL - https://finance.yahoo.com/quote/XEL
80. EIX - https://finance.yahoo.com/quote/EIX
81. T - https://finance.yahoo.com/quote/T
82. VZ - https://finance.yahoo.com/quote/VZ
83. TMUS - https://finance.yahoo.com/quote/TMUS
84. CMCSA - https://finance.yahoo.com/quote/CMCSA
85. CHTR - https://finance.yahoo.com/quote/CHTR
86. VOD - https://finance.yahoo.com/quote/VOD
87. BCE - https://finance.yahoo.com/quote/BCE
88. TU - https://finance.yahoo.com/quote/TU
89. AMT - https://finance.yahoo.com/quote/AMT
90. SBAC - https://finance.yahoo.com/quote/SBAC
91. BHP - https://finance.yahoo.com/quote/BHP
92. RIO - https://finance.yahoo.com/quote/RIO
93. VALE - https://finance.yahoo.com/quote/VALE
94. FCX - https://finance.yahoo.com/quote/FCX
95. NEM - https://finance.yahoo.com/quote/NEM
96. APD - https://finance.yahoo.com/quote/APD
97. ECL - https://finance.yahoo.com/quote/ECL
98. SHW - https://finance.yahoo.com/quote/SHW
99. LYB - https://finance.yahoo.com/quote/LYB
100. DD - https://finance.yahoo.com/quote/DD

bonds

1. AGG - https://finance.yahoo.com/quote/AGG  
2. BND - https://finance.yahoo.com/quote/BND  
3. TLT - https://finance.yahoo.com/quote/TLT  
4. IEF - https://finance.yahoo.com/quote/IEF  
5. LQD - https://finance.yahoo.com/quote/LQD  
6. HYG - https://finance.yahoo.com/quote/HYG  
7. TIP - https://finance.yahoo.com/quote/TIP  
8. SHY - https://finance.yahoo.com/quote/SHY  
9. EMB - https://finance.yahoo.com/quote/EMB  
10. MUB - https://finance.yahoo.com/quote/MUB  
11. BIV - https://finance.yahoo.com/quote/BIV  
12. BSV - https://finance.yahoo.com/quote/BSV  
13. VCIT - https://finance.yahoo.com/quote/VCIT  
14. VCLT - https://finance.yahoo.com/quote/VCLT  
15. BNDX - https://finance.yahoo.com/quote/BNDX  
16. BWX - https://finance.yahoo.com/quote/BWX  
17. CWB - https://finance.yahoo.com/quote/CWB  
18. JNK - https://finance.yahoo.com/quote/JNK  
19. HYD - https://finance.yahoo.com/quote/HYD  
20. BAB - https://finance.yahoo.com/quote/BAB  

commodities

1. CL=F - https://finance.yahoo.com/quote/CL=F
2. BZ=F - https://finance.yahoo.com/quote/BZ=F
3. NG=F - https://finance.yahoo.com/quote/NG=F
4. GC=F - https://finance.yahoo.com/quote/GC=F
5. SI=F - https://finance.yahoo.com/quote/SI=F
6. PL=F - https://finance.yahoo.com/quote/PL=F
7. PA=F - https://finance.yahoo.com/quote/PA=F
8. HG=F - https://finance.yahoo.com/quote/HG=F

[Дополнительный датасет содержит информацию о валютах].
