# FAISS Evaluation Report - faiss_eval_2025-05-25_k10_e5_m+

## Configuration
- `Note: Improved FAISS index adding 'Nombre Medicamento: ' at the beginning of each chunk`
- `Top-k`: 10
- `Embedding model`: `intfloat/multilingual-e5-large`
- `Device`: `cuda`
- `Date`: 2025-05-25 08:57

## Global Results

- **Mean precision@k**: 0.1886
- **Mean normalized precision@k**: 0.3182
- **Mean recall@k**: 0.2797
- **Mean hit@k**: 0.57
- **Mean mrr**: 0.3265
- **Mean n relevant retrieved**: 1.89
- **Questions evaluated**: 2630
- **K**: 10
- **Embedding model**: intfloat/multilingual-e5-large
- **Device**: cuda
- **Run name**: faiss_eval_2025-05-25_k10_e5_m+

---
## Sample Questions (Top 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['3175 ESP', '810 ESP', '3175 ESP', '3776 ESP', '4023 ESP', '3656 ESP', '3175 ESP', '2157 ESP', '3656 ESP', '4076 ESP']
  - P@10: 0.5, R@10: 0.625, MRR: 0.25

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['810 ESP', '3806 ESP', '3804 ESP', '1310 ESP', '4008 ESP', '4009 ESP', '690 ESP', '4009 ESP', '3805 ESP', '4008 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['3733 ESP', '2519 ESP', '3700 ESP', '4202 ESP', '3010 ESP', '4384 ESP', '2519 ESP', '3700 ESP', '4401 ESP', '792 ESP']
  - P@10: 0.2, R@10: 0.6666666666666666, MRR: 0.25

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3128 ESP', '3569 ESP', '3274 ESP', '1860 ESP', '2228 ESP', '2029 ESP', '3629 ESP', '971 ESP', '3152 ESP', '3458 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1680 ESP', '4081 ESP', '1848 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '2335 ESP', '1680 ESP', '1680 ESP', '1680 ESP']
  - P@10: 0.0, R@10: 0.0, MRR: 0.0

