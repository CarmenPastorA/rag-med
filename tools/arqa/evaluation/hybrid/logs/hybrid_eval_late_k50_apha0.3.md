# Hybrid Evaluation Report (late fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 50
- Final top-K (for metrics): 50
- Alpha (fusion weight): 0.3
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.1099
- **mean_normalized_precision@k**: 0.673
- **mean_recall@k**: 0.6677
- **mean_hit@k**: 0.9011
- **mean_mrr**: 0.2375
- **mean_n_relevant_retrieved**: 5.5
- **questions_evaluated**: 2630
- **mode**: late
- **bm25_top_n**: 50
- **faiss_top_k**: 50
- **final_top_k**: 50
- **alpha**: 0.3
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['2630 ESP', '1223 ESP', '3656 ESP', '3331 ESP', '1654 ESP', '2621 ESP', '2665 ESP', '944 ESP', '3697 ESP', '14 ESP', '810 ESP', '980 ESP', '570 ESP', '1053 ESP', '4076 ESP', '2289 ESP', '1131 ESP', '1222 ESP', '3380 ESP', '3972 ESP', '4023 ESP', '3768 ESP', '2688 ESP', '3776 ESP', '1655 ESP', '1071 ESP', '3451 ESP', '903 ESP', '2671 ESP', '2065 ESP', '13 ESP', '1068 ESP', '590 ESP', '2260 ESP', '2265 ESP', '3034 ESP', '2852 ESP', '3151 ESP', '1221 ESP', '3003 ESP', '3175 ESP', '3487 ESP', '2319 ESP', '2531 ESP', '12 ESP', '3513 ESP', '2277 ESP', '942 ESP', '4239 ESP', '2157 ESP']
  - P@50: 0.16, R@50: 1.0, MRR: 1.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['1310 ESP', '2574 ESP', '4334 ESP', '1971 ESP', '1654 ESP', '4203 ESP', '2757 ESP', '3820 ESP', '3395 ESP', '4332 ESP', '690 ESP', '3126 ESP', '1149 ESP', '3758 ESP', '327 ESP', '2749 ESP', '3804 ESP', '3393 ESP', '1066 ESP', '2284 ESP', '2416 ESP', '2750 ESP', '4010 ESP', '4333 ESP', '4321 ESP', '2756 ESP', '3985 ESP', '2222 ESP', '4398 ESP', '3627 ESP', '3819 ESP', '3805 ESP', '3023 ESP', '1435 ESP', '4009 ESP', '2758 ESP', '3620 ESP', '3968 ESP', '4008 ESP', '4011 ESP', '1 ESP', '3983 ESP', '3582 ESP', '3806 ESP', '3458 ESP', '3984 ESP', '3158 ESP', '2 ESP', '2873 ESP', '3516 ESP']
  - P@50: 0.06, R@50: 0.42857142857142855, MRR: 0.3333333333333333

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['1310 ESP', '2277 ESP', '4202 ESP', '792 ESP', '3365 ESP', '1506 ESP', '2648 ESP', '2315 ESP', '3881 ESP', '3619 ESP', '2431 ESP', '1518 ESP', '2223 ESP', '4127 ESP', '1519 ESP', '3700 ESP', '3715 ESP', '4114 ESP', '4401 ESP', '3683 ESP', '816 ESP', '3129 ESP', '2921 ESP', '3008 ESP', '3502 ESP', '3733 ESP', '2881 ESP', '2221 ESP', '1952 ESP', '3399 ESP', '3010 ESP', '2519 ESP', '3615 ESP', '4406 ESP', '2222 ESP', '3842 ESP', '4113 ESP', '4384 ESP', '869 ESP', '3865 ESP', '2282 ESP', '2716 ESP', '3778 ESP', '1602 ESP', '3130 ESP', '2001 ESP', '3844 ESP', '3454 ESP', '3453 ESP', '1892 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.3333333333333333

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3128 ESP', '3152 ESP', '1235 ESP', '4 ESP', '1931 ESP', '811 ESP', '1654 ESP', '4083 ESP', '4239 ESP', '4379 ESP', '3697 ESP', '14 ESP', '2220 ESP', '2582 ESP', '259 ESP', '3768 ESP', '3776 ESP', '3204 ESP', '3629 ESP', '2029 ESP', '1655 ESP', '4216 ESP', '431 ESP', '3451 ESP', '3400 ESP', '2671 ESP', '2065 ESP', '4321 ESP', '971 ESP', '2048 ESP', '4398 ESP', '965 ESP', '1068 ESP', '13 ESP', '647 ESP', '3569 ESP', '966 ESP', '1860 ESP', '3034 ESP', '3151 ESP', '3003 ESP', '2228 ESP', '402 ESP', '3160 ESP', '2739 ESP', '3458 ESP', '1242 ESP', '2277 ESP', '1298 ESP', '1771 ESP']
  - P@50: 0.06, R@50: 0.6, MRR: 0.1111111111111111

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1325 ESP', '3466 ESP', '2574 ESP', '3424 ESP', '2335 ESP', '4301 ESP', '3679 ESP', '1235 ESP', '3331 ESP', '3976 ESP', '1873 ESP', '2385 ESP', '1979 ESP', '4083 ESP', '2146 ESP', '3392 ESP', '1796 ESP', '570 ESP', '1794 ESP', '2011 ESP', '1732 ESP', '1879 ESP', '3204 ESP', '1195 ESP', '1848 ESP', '1066 ESP', '2479 ESP', '1980 ESP', '3451 ESP', '370 ESP', '1733 ESP', '4321 ESP', '4398 ESP', '361 ESP', '1314 ESP', '2339 ESP', '571 ESP', '2260 ESP', '1850 ESP', '3242 ESP', '2052 ESP', '2132 ESP', '1718 ESP', '1646 ESP', '2012 ESP', '2478 ESP', '3338 ESP', '858 ESP', '3184 ESP', '2576 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.05555555555555555

