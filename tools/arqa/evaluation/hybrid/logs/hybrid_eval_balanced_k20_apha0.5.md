# Hybrid Evaluation Report (balanced fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 50
- Final top-K (for metrics): 20
- Alpha (fusion weight): 0.5
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.1521
- **mean_normalized_precision@k**: 0.3973
- **mean_recall@k**: 0.3799
- **mean_hit@k**: 0.7316
- **mean_mrr**: 0.2771
- **mean_n_relevant_retrieved**: 3.04
- **questions_evaluated**: 2630
- **mode**: balanced
- **bm25_top_n**: 50
- **faiss_top_k**: 50
- **final_top_k**: 20
- **alpha**: 0.5
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['1071 ESP', '1222 ESP', '1068 ESP', '570 ESP', '3380 ESP', '1131 ESP', '2630 ESP', '4239 ESP', '2277 ESP', '1654 ESP', '2289 ESP', '3776 ESP', '2671 ESP', '12 ESP', '14 ESP', '13 ESP', '3513 ESP', '903 ESP', '3034 ESP', '3451 ESP']
  - P@20: 0.2, R@20: 0.5, MRR: 0.14285714285714285

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['3627 ESP', '2757 ESP', '2756 ESP', '4009 ESP', '4008 ESP', '4321 ESP', '1066 ESP', '2758 ESP', '327 ESP', '1 ESP', '3620 ESP', '2 ESP', '3984 ESP', '3806 ESP', '4010 ESP', '3758 ESP', '3805 ESP', '1149 ESP', '4011 ESP', '3804 ESP']
  - P@20: 0.0, R@20: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2282 ESP', '3733 ESP', '4113 ESP', '4114 ESP', '3683 ESP', '3715 ESP', '3129 ESP', '3865 ESP', '2315 ESP', '3008 ESP', '1952 ESP', '3365 ESP', '4406 ESP', '1519 ESP', '4202 ESP', '2648 ESP', '1518 ESP', '1602 ESP', '3010 ESP', '869 ESP']
  - P@20: 0.05, R@20: 0.3333333333333333, MRR: 0.06666666666666667

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['1242 ESP', '1068 ESP', '4216 ESP', '3151 ESP', '4239 ESP', '2277 ESP', '1654 ESP', '4379 ESP', '3152 ESP', '3776 ESP', '2671 ESP', '13 ESP', '3160 ESP', '3400 ESP', '1655 ESP', '4083 ESP', '4398 ESP', '2220 ESP', '3034 ESP', '3451 ESP']
  - P@20: 0.15, R@20: 0.6, MRR: 0.2

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['2011 ESP', '3242 ESP', '3679 ESP', '2478 ESP', '1718 ESP', '1979 ESP', '1873 ESP', '1235 ESP', '2339 ESP', '2012 ESP', '2146 ESP', '2132 ESP', '4083 ESP', '2052 ESP', '4398 ESP', '2335 ESP', '3338 ESP', '361 ESP', '858 ESP', '3451 ESP']
  - P@20: 0.0, R@20: 0.0, MRR: 0.0

