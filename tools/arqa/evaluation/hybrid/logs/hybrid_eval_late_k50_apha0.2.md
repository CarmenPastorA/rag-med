# Hybrid Evaluation Report (late fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 50
- Final top-K (for metrics): 50
- Alpha (fusion weight): 0.2
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.1088
- **mean_normalized_precision@k**: 0.667
- **mean_recall@k**: 0.6617
- **mean_hit@k**: 0.9004
- **mean_mrr**: 0.2393
- **mean_n_relevant_retrieved**: 5.44
- **questions_evaluated**: 2630
- **mode**: late
- **bm25_top_n**: 50
- **faiss_top_k**: 50
- **final_top_k**: 50
- **alpha**: 0.2
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['2630 ESP', '570 ESP', '1223 ESP', '14 ESP', '2065 ESP', '810 ESP', '3697 ESP', '2852 ESP', '3122 ESP', '980 ESP', '2289 ESP', '4239 ESP', '2319 ESP', '3331 ESP', '1131 ESP', '903 ESP', '1071 ESP', '3151 ESP', '3175 ESP', '13 ESP', '3487 ESP', '3768 ESP', '1053 ESP', '2157 ESP', '2665 ESP', '3513 ESP', '2409 ESP', '3776 ESP', '12 ESP', '3656 ESP', '3003 ESP', '1221 ESP', '3972 ESP', '1068 ESP', '2671 ESP', '3034 ESP', '2688 ESP', '2621 ESP', '2277 ESP', '942 ESP', '3451 ESP', '4076 ESP', '571 ESP', '2265 ESP', '3380 ESP', '1222 ESP', '590 ESP', '2260 ESP', '2531 ESP', '4023 ESP']
  - P@50: 0.16, R@50: 1.0, MRR: 1.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['3516 ESP', '3582 ESP', '4231 ESP', '3395 ESP', '2 ESP', '327 ESP', '3158 ESP', '4009 ESP', '2873 ESP', '1 ESP', '3458 ESP', '4334 ESP', '810 ESP', '2416 ESP', '4010 ESP', '2749 ESP', '3804 ESP', '690 ESP', '2756 ESP', '3820 ESP', '3985 ESP', '3627 ESP', '1310 ESP', '2574 ESP', '3758 ESP', '3983 ESP', '552 ESP', '3968 ESP', '2750 ESP', '4203 ESP', '2758 ESP', '4011 ESP', '3620 ESP', '4332 ESP', '3984 ESP', '1066 ESP', '4321 ESP', '4398 ESP', '4008 ESP', '2757 ESP', '1149 ESP', '2222 ESP', '3126 ESP', '4333 ESP', '1654 ESP', '3806 ESP', '1435 ESP', '3805 ESP', '3393 ESP', '3819 ESP']
  - P@50: 0.06, R@50: 0.42857142857142855, MRR: 0.08333333333333333

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2001 ESP', '4114 ESP', '1602 ESP', '3615 ESP', '3865 ESP', '1506 ESP', '1518 ESP', '2223 ESP', '3844 ESP', '2413 ESP', '3700 ESP', '2881 ESP', '3365 ESP', '4406 ESP', '3502 ESP', '2648 ESP', '1892 ESP', '2282 ESP', '3778 ESP', '1519 ESP', '2466 ESP', '1310 ESP', '3733 ESP', '4384 ESP', '2716 ESP', '2519 ESP', '2221 ESP', '816 ESP', '4202 ESP', '869 ESP', '1986 ESP', '792 ESP', '2431 ESP', '3010 ESP', '2921 ESP', '3715 ESP', '4127 ESP', '2222 ESP', '3008 ESP', '1952 ESP', '2315 ESP', '4401 ESP', '3399 ESP', '3881 ESP', '3842 ESP', '3129 ESP', '3130 ESP', '4298 ESP', '3683 ESP', '4113 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.034482758620689655

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['402 ESP', '3204 ESP', '2029 ESP', '2739 ESP', '2065 ESP', '3458 ESP', '14 ESP', '3697 ESP', '2582 ESP', '4239 ESP', '971 ESP', '966 ESP', '811 ESP', '431 ESP', '1235 ESP', '2290 ESP', '3152 ESP', '2287 ESP', '13 ESP', '2048 ESP', '3400 ESP', '3768 ESP', '4216 ESP', '3569 ESP', '1931 ESP', '4 ESP', '1242 ESP', '3629 ESP', '3003 ESP', '4398 ESP', '1655 ESP', '1068 ESP', '2671 ESP', '259 ESP', '3274 ESP', '3160 ESP', '3034 ESP', '1654 ESP', '4379 ESP', '2277 ESP', '647 ESP', '3451 ESP', '3128 ESP', '1860 ESP', '965 ESP', '1771 ESP', '1298 ESP', '3776 ESP', '2228 ESP', '4083 ESP']
  - P@50: 0.06, R@50: 0.6, MRR: 0.1

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['4301 ESP', '2478 ESP', '570 ESP', '3204 ESP', '3392 ESP', '3338 ESP', '2385 ESP', '1718 ESP', '3976 ESP', '3679 ESP', '2132 ESP', '3331 ESP', '361 ESP', '1235 ESP', '2339 ESP', '2479 ESP', '2574 ESP', '1732 ESP', '2146 ESP', '1794 ESP', '1796 ESP', '1314 ESP', '1646 ESP', '1848 ESP', '1980 ESP', '1680 ESP', '2335 ESP', '1979 ESP', '4398 ESP', '1066 ESP', '4321 ESP', '1879 ESP', '1873 ESP', '858 ESP', '2012 ESP', '1850 ESP', '3242 ESP', '2052 ESP', '3424 ESP', '2576 ESP', '3451 ESP', '1733 ESP', '571 ESP', '3184 ESP', '3466 ESP', '1195 ESP', '2011 ESP', '2260 ESP', '4083 ESP', '1325 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.3333333333333333

