# Hierarchical FAISS Evaluation - hierarchical_faiss_eval_doc50_chunk10

## Configuration
- Doc top-N: 50
- Chunk top-K: 10
- Device: cuda

## Global Results

- **mean_precision@k**: 0.2105
- **mean_normalized_precision@k**: 0.3598
- **mean_recall@k**: 0.3178
- **mean_hit@k**: 0.5992
- **mean_mrr**: 0.3512
- **mean_n_relevant_retrieved**: 2.11
- **questions_evaluated**: 2630
- **doc_top_n**: 50
- **chunk_top_k**: 10
- **device**: cuda

---
## Sample Questions (first 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['3175 ESP', '810 ESP', '3175 ESP', '3776 ESP', '4023 ESP', '3656 ESP', '3175 ESP', '2157 ESP', '3656 ESP', '4076 ESP']
  - P@10: 0.5, R@10: 0.625, MRR: 0.25

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['3804 ESP', '4008 ESP', '4009 ESP', '690 ESP', '4009 ESP', '4008 ESP', '3627 ESP', '4334 ESP', '3395 ESP', '3804 ESP']
  - P@10: 0.1, R@10: 0.14285714285714285, MRR: 0.125

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2519 ESP', '4202 ESP', '2519 ESP', '4401 ESP', '816 ESP', '1892 ESP', '2223 ESP', '1892 ESP', '2921 ESP', '3399 ESP']
  - P@10: 0.2, R@10: 0.6666666666666666, MRR: 0.5

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3128 ESP', '3569 ESP', '3274 ESP', '2228 ESP', '2029 ESP', '971 ESP', '3152 ESP', '3458 ESP', '3204 ESP', '431 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1680 ESP', '4081 ESP', '1848 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '2335 ESP', '1680 ESP', '1680 ESP', '1680 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

