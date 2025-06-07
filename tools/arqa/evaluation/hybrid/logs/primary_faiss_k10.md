# Primary FAISS Evaluation - primary_faiss_k10

## Configuration
- Doc top-N: 10
- Device: cuda

## Global Results

- **mean_precision@k**: 0.1976
- **mean_normalized_precision@k**: 0.3356
- **mean_recall@k**: 0.299
- **mean_hit@k**: 0.6354
- **mean_mrr**: 0.3958
- **mean_n_relevant_retrieved**: 1.98
- **questions_evaluated**: 2630
- **doc_top_n**: 10
- **device**: cuda

---
## Sample Questions (first 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['4076 ESP', '3656 ESP', '3768 ESP', '2157 ESP', '3776 ESP', '14 ESP', '3152 ESP', '1053 ESP', '810 ESP', '821 ESP']
  - P@10: 0.6, R@10: 0.75, MRR: 1.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['4332 ESP', '4333 ESP', '4008 ESP', '4009 ESP', '3393 ESP', '4334 ESP', '3395 ESP', '638 ESP', '4013 ESP', '4012 ESP']
  - P@10: 0.3, R@10: 0.42857142857142855, MRR: 1.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2921 ESP', '4174 ESP', '1071 ESP', '816 ESP', '2994 ESP', '2519 ESP', '3825 ESP', '2466 ESP', '3092 ESP', '4401 ESP']
  - P@10: 0.1, R@10: 0.3333333333333333, MRR: 0.1

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['2228 ESP', '3656 ESP', '971 ESP', '4076 ESP', '2157 ESP', '2029 ESP', '3152 ESP', '3768 ESP', '3003 ESP', '3346 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['821 ESP', '571 ESP', '570 ESP', '402 ESP', '2223 ESP', '1646 ESP', '2222 ESP', '1796 ESP', '2288 ESP', '1680 ESP']
  - P@10: 0.3, R@10: 1.0, MRR: 1.0

