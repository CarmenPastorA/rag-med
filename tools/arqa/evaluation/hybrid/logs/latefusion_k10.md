# Hybrid Evaluation Report (late fusion)

## Configuration
- BM25 top-N: 50
- FAISS top-K: 50
- Final top-K (for metrics): 10
- Alpha (fusion weight): 0.5
- Device: cuda

## Global Metrics
- **mean_precision@k**: 0.2371
- **mean_normalized_precision@k**: 0.3951
- **mean_recall@k**: 0.3495
- **mean_hit@k**: 0.7335
- **mean_mrr**: 0.3939
- **mean_n_relevant_retrieved**: 2.37
- **questions_evaluated**: 2630
- **mode**: late
- **bm25_top_n**: 50
- **faiss_top_k**: 50
- **final_top_k**: 10
- **alpha**: 0.5
- **device**: cuda
- **fallback_count**: 0
- **fallback_ratio**: 0.0

---
## Sample Questions (First 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['2665 ESP', '2289 ESP', '3656 ESP', '1068 ESP', '1053 ESP', '3776 ESP', '3768 ESP', '810 ESP', '2157 ESP', '14 ESP']
  - P@10: 0.5, R@10: 0.625, MRR: 0.3333333333333333

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['3805 ESP', '4009 ESP', '3804 ESP', '3984 ESP', '3395 ESP', '3985 ESP', '3627 ESP', '4321 ESP', '4008 ESP', '3806 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['4113 ESP', '4114 ESP', '3010 ESP', '1602 ESP', '4401 ESP', '4202 ESP', '1519 ESP', '3733 ESP', '1518 ESP', '4384 ESP']
  - P@10: 0.2, R@10: 0.6666666666666666, MRR: 0.2

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3204 ESP', '2277 ESP', '1068 ESP', '3128 ESP', '3776 ESP', '4398 ESP', '1242 ESP', '3451 ESP', '3034 ESP', '3569 ESP']
  - P@10: 0.1, R@10: 0.2, MRR: 0.1111111111111111

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['3679 ESP', '3338 ESP', '2132 ESP', '2339 ESP', '1718 ESP', '2335 ESP', '4083 ESP', '361 ESP', '3451 ESP', '2146 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

