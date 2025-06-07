# Hybrid Evaluation Report (late fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 50
- Final top-K (for metrics): 50
- Alpha (fusion weight): 0.5
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.1113
- **mean_normalized_precision@k**: 0.679
- **mean_recall@k**: 0.6737
- **mean_hit@k**: 0.8985
- **mean_mrr**: 0.2581
- **mean_n_relevant_retrieved**: 5.56
- **questions_evaluated**: 2630
- **mode**: late
- **bm25_top_n**: 50
- **faiss_top_k**: 50
- **final_top_k**: 50
- **alpha**: 0.5
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['4076 ESP', '2688 ESP', '3175 ESP', '3697 ESP', '2665 ESP', '14 ESP', '4023 ESP', '3034 ESP', '2671 ESP', '1221 ESP', '2289 ESP', '3487 ESP', '2265 ESP', '2621 ESP', '2260 ESP', '3776 ESP', '570 ESP', '3003 ESP', '590 ESP', '2157 ESP', '2630 ESP', '1648 ESP', '3451 ESP', '13 ESP', '12 ESP', '2277 ESP', '980 ESP', '903 ESP', '1131 ESP', '944 ESP', '4239 ESP', '3380 ESP', '1222 ESP', '2220 ESP', '4216 ESP', '1071 ESP', '2852 ESP', '1053 ESP', '2290 ESP', '2319 ESP', '810 ESP', '3513 ESP', '2065 ESP', '1655 ESP', '1068 ESP', '3768 ESP', '942 ESP', '3656 ESP', '580 ESP', '1654 ESP']
  - P@50: 0.16, R@50: 1.0, MRR: 1.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['4010 ESP', '3458 ESP', '4357 ESP', '1066 ESP', '3393 ESP', '1971 ESP', '3819 ESP', '3806 ESP', '2873 ESP', '2285 ESP', '3627 ESP', '3395 ESP', '2 ESP', '3805 ESP', '2758 ESP', '3158 ESP', '4332 ESP', '4009 ESP', '3126 ESP', '327 ESP', '3968 ESP', '4334 ESP', '3983 ESP', '2284 ESP', '2173 ESP', '4345 ESP', '2750 ESP', '3620 ESP', '2416 ESP', '3984 ESP', '1310 ESP', '1 ESP', '3985 ESP', '3820 ESP', '2757 ESP', '4011 ESP', '3804 ESP', '3516 ESP', '1435 ESP', '4008 ESP', '4321 ESP', '2749 ESP', '2756 ESP', '1149 ESP', '2222 ESP', '3758 ESP', '4398 ESP', '3023 ESP', '1654 ESP', '690 ESP']
  - P@50: 0.04, R@50: 0.2857142857142857, MRR: 0.058823529411764705

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['3352 ESP', '2921 ESP', '2519 ESP', '3970 ESP', '3010 ESP', '3283 ESP', '3454 ESP', '4406 ESP', '2431 ESP', '3715 ESP', '3881 ESP', '4114 ESP', '3399 ESP', '1602 ESP', '2223 ESP', '1518 ESP', '3008 ESP', '3453 ESP', '2221 ESP', '1506 ESP', '3615 ESP', '3130 ESP', '2277 ESP', '3365 ESP', '1892 ESP', '4113 ESP', '4127 ESP', '3865 ESP', '3619 ESP', '1310 ESP', '2282 ESP', '4202 ESP', '1952 ESP', '4239 ESP', '816 ESP', '3844 ESP', '32 ESP', '4401 ESP', '1519 ESP', '3683 ESP', '2648 ESP', '3733 ESP', '3129 ESP', '3636 ESP', '2222 ESP', '792 ESP', '2315 ESP', '3842 ESP', '869 ESP', '4384 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.03125

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['4379 ESP', '1931 ESP', '3697 ESP', '3458 ESP', '3151 ESP', '3034 ESP', '2671 ESP', '3569 ESP', '1298 ESP', '965 ESP', '2029 ESP', '3776 ESP', '3003 ESP', '3160 ESP', '431 ESP', '3451 ESP', '13 ESP', '402 ESP', '1860 ESP', '2277 ESP', '971 ESP', '903 ESP', '1242 ESP', '647 ESP', '966 ESP', '944 ESP', '4239 ESP', '4136 ESP', '4216 ESP', '4083 ESP', '2220 ESP', '2582 ESP', '811 ESP', '3204 ESP', '2228 ESP', '2048 ESP', '4321 ESP', '3400 ESP', '3629 ESP', '4128 ESP', '4135 ESP', '1655 ESP', '1068 ESP', '3768 ESP', '1775 ESP', '4398 ESP', '3128 ESP', '1771 ESP', '1654 ESP', '2739 ESP']
  - P@50: 0.1, R@50: 1.0, MRR: 0.16666666666666666

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1796 ESP', '858 ESP', '2478 ESP', '3466 ESP', '2385 ESP', '1732 ESP', '2011 ESP', '1066 ESP', '1325 ESP', '1794 ESP', '2339 ESP', '1235 ESP', '2260 ESP', '3424 ESP', '3331 ESP', '570 ESP', '2576 ESP', '1718 ESP', '571 ESP', '2335 ESP', '1848 ESP', '1733 ESP', '2574 ESP', '1980 ESP', '3451 ESP', '361 ESP', '3681 ESP', '370 ESP', '3184 ESP', '1879 ESP', '1195 ESP', '3976 ESP', '1646 ESP', '2132 ESP', '4083 ESP', '3392 ESP', '3679 ESP', '1873 ESP', '3204 ESP', '1979 ESP', '3338 ESP', '3242 ESP', '4321 ESP', '2146 ESP', '4301 ESP', '2479 ESP', '1314 ESP', '2012 ESP', '2052 ESP', '4398 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.0625

