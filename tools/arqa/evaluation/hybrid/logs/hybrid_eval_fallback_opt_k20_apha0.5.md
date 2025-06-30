# Hybrid Evaluation Report (fallback_opt fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 20
- Final top-K (for metrics): 20
- Alpha (fusion weight): 0.5
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.1709
- **mean_normalized_precision@k**: 0.966
- **mean_recall@k**: 0.9309
- **mean_hit@k**: 0.8536
- **mean_mrr**: 0.341
- **mean_n_relevant_retrieved**: 6.83
- **questions_evaluated**: 2630
- **mode**: fallback_opt
- **bm25_top_n**: 50
- **faiss_top_k**: 20
- **final_top_k**: 20
- **alpha**: 0.5
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['3175 ESP', '810 ESP', '3175 ESP', '3776 ESP', '4023 ESP', '3656 ESP', '3175 ESP', '2157 ESP', '3656 ESP', '4076 ESP', '3768 ESP', '1221 ESP', '2621 ESP', '3656 ESP', '2157 ESP', '2665 ESP', '1053 ESP', '4076 ESP', '2260 ESP', '14 ESP', '13 ESP', '13 ESP', '2671 ESP', '2671 ESP', '2277 ESP', '2277 ESP', '1068 ESP', '1068 ESP', '1071 ESP', '1071 ESP', '4239 ESP', '4239 ESP', '2630 ESP', '2630 ESP', '3451 ESP', '3451 ESP', '3034 ESP', '3034 ESP', '1131 ESP', '1131 ESP']
  - P@20: 0.35, R@20: 1.75, MRR: 0.25

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['810 ESP', '3806 ESP', '3804 ESP', '1310 ESP', '4008 ESP', '4009 ESP', '690 ESP', '4009 ESP', '3805 ESP', '4008 ESP', '3627 ESP', '1310 ESP', '3458 ESP', '3985 ESP', '4203 ESP', '3516 ESP', '3968 ESP', '3820 ESP', '4334 ESP', '3395 ESP', '2756 ESP', '2756 ESP', '2758 ESP', '2758 ESP', '2757 ESP', '2757 ESP', '1066 ESP', '1066 ESP', '3620 ESP', '3620 ESP', '3758 ESP', '3758 ESP', '1149 ESP', '1149 ESP', '327 ESP', '327 ESP', '1 ESP', '1 ESP', '2 ESP', '2 ESP']
  - P@20: 0.025, R@20: 0.14285714285714285, MRR: 0.05263157894736842

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['3733 ESP', '2519 ESP', '3700 ESP', '4202 ESP', '3010 ESP', '4384 ESP', '2519 ESP', '3700 ESP', '4401 ESP', '792 ESP', '816 ESP', '3842 ESP', '1892 ESP', '2223 ESP', '2221 ESP', '4127 ESP', '1892 ESP', '2413 ESP', '3881 ESP', '2921 ESP', '1518 ESP', '1518 ESP', '1602 ESP', '1602 ESP', '1519 ESP', '1519 ESP', '4113 ESP', '4113 ESP', '4114 ESP', '4114 ESP', '3683 ESP', '3683 ESP', '869 ESP', '869 ESP', '2315 ESP', '2315 ESP', '3008 ESP', '3008 ESP', '3365 ESP', '3365 ESP']
  - P@20: 0.05, R@20: 0.6666666666666666, MRR: 0.25

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3128 ESP', '3569 ESP', '3274 ESP', '1860 ESP', '2228 ESP', '2029 ESP', '3629 ESP', '971 ESP', '3152 ESP', '3458 ESP', '3204 ESP', '811 ESP', '2277 ESP', '3697 ESP', '431 ESP', '1298 ESP', '2582 ESP', '647 ESP', '3003 ESP', '966 ESP', '3451 ESP', '3451 ESP', '1242 ESP', '1242 ESP', '4398 ESP', '4398 ESP', '2671 ESP', '2671 ESP', '3776 ESP', '3776 ESP', '1654 ESP', '1654 ESP', '1655 ESP', '1655 ESP', '4216 ESP', '4216 ESP', '3034 ESP', '3034 ESP', '13 ESP', '13 ESP']
  - P@20: 0.1, R@20: 0.8, MRR: 0.02702702702702703

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1680 ESP', '4081 ESP', '1848 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '2335 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '3166 ESP', '4081 ESP', '4081 ESP', '1794 ESP', '1325 ESP', '1796 ESP', '2576 ESP', '1680 ESP', '3166 ESP', '4083 ESP', '4083 ESP', '2146 ESP', '2146 ESP', '1718 ESP', '1718 ESP', '2132 ESP', '2132 ESP', '361 ESP', '361 ESP', '3451 ESP', '3451 ESP', '3338 ESP', '3338 ESP', '3679 ESP', '3679 ESP', '2339 ESP', '2339 ESP', '2012 ESP', '2012 ESP']
  - P@20: 0.0, R@20: 0.0, MRR: 0.0

