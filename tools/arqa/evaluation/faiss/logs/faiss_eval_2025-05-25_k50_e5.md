# FAISS Evaluation Report - faiss_eval_2025-05-25_k50_e5

## Configuration
- `Note: Improved FAISS index adding 'Nombre Medicamento: ' at the beginning of each chunk`
- `Top-k`: 50
- `Embedding model`: `intfloat/multilingual-e5-large`
- `Device`: `cuda`
- `Date`: 2025-05-25 12:17

## Global Results

- **Mean normalized precision@k**: 0.842
- **Mean recall@k**: 0.8363
- **Mean hit@k**: 0.7962
- **Mean mrr**: 0.3376
- **Mean n relevant retrieved**: 6.45
- **Questions evaluated**: 2630
- **K**: 50
- **Embedding model**: intfloat/multilingual-e5-large
- **Device**: cuda
- **Run name**: faiss_eval_2025-05-25_k50_e5

---
## Sample Questions (Top 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['3175 ESP', '810 ESP', '3175 ESP', '3776 ESP', '4023 ESP', '3656 ESP', '3175 ESP', '2157 ESP', '3656 ESP', '4076 ESP', '3768 ESP', '1221 ESP', '2621 ESP', '3656 ESP', '2157 ESP', '2665 ESP', '1053 ESP', '4076 ESP', '2260 ESP', '14 ESP', '2289 ESP', '2688 ESP', '3003 ESP', '590 ESP', '3487 ESP', '980 ESP', '3697 ESP', '3175 ESP', '2852 ESP', '942 ESP', '3487 ESP', '2065 ESP', '1068 ESP', '2265 ESP', '2319 ESP', '3151 ESP', '1223 ESP', '2531 ESP', '3972 ESP', '3331 ESP', '3122 ESP', '571 ESP', '1068 ESP', '2409 ESP', '3749 ESP', '2987 ESP', '2628 ESP', '3776 ESP', '2451 ESP', '4076 ESP']
  - P@50: 0.24, R@50: 1.5, MRR: 0.25

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['810 ESP', '3806 ESP', '3804 ESP', '1310 ESP', '4008 ESP', '4009 ESP', '690 ESP', '4009 ESP', '3805 ESP', '4008 ESP', '3627 ESP', '1310 ESP', '3458 ESP', '3985 ESP', '4203 ESP', '3516 ESP', '3968 ESP', '3820 ESP', '4334 ESP', '3395 ESP', '3804 ESP', '3393 ESP', '3984 ESP', '4321 ESP', '3819 ESP', '3158 ESP', '1435 ESP', '2222 ESP', '3393 ESP', '2750 ESP', '4010 ESP', '3126 ESP', '4332 ESP', '2873 ESP', '810 ESP', '2220 ESP', '4203 ESP', '4203 ESP', '2574 ESP', '3983 ESP', '4333 ESP', '3582 ESP', '810 ESP', '4231 ESP', '552 ESP', '2223 ESP', '2872 ESP', '2749 ESP', '4011 ESP', '2220 ESP']
  - P@50: 0.06, R@50: 0.42857142857142855, MRR: 0.05263157894736842

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['3733 ESP', '2519 ESP', '3700 ESP', '4202 ESP', '3010 ESP', '4384 ESP', '2519 ESP', '3700 ESP', '4401 ESP', '792 ESP', '816 ESP', '3842 ESP', '1892 ESP', '2223 ESP', '2221 ESP', '4127 ESP', '1892 ESP', '2413 ESP', '3881 ESP', '2921 ESP', '2716 ESP', '3399 ESP', '2222 ESP', '2921 ESP', '1310 ESP', '2413 ESP', '3615 ESP', '3844 ESP', '4030 ESP', '2431 ESP', '1986 ESP', '2519 ESP', '2001 ESP', '3700 ESP', '3502 ESP', '2881 ESP', '2001 ESP', '3778 ESP', '2716 ESP', '2466 ESP', '4298 ESP', '2466 ESP', '1986 ESP', '2413 ESP', '3744 ESP', '4030 ESP', '3642 ESP', '2076 ESP', '3958 ESP', '3374 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.25

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['3128 ESP', '3569 ESP', '3274 ESP', '1860 ESP', '2228 ESP', '2029 ESP', '3629 ESP', '971 ESP', '3152 ESP', '3458 ESP', '3204 ESP', '811 ESP', '2277 ESP', '3697 ESP', '431 ESP', '1298 ESP', '2582 ESP', '647 ESP', '3003 ESP', '966 ESP', '965 ESP', '4076 ESP', '2048 ESP', '2739 ESP', '3656 ESP', '1068 ESP', '3768 ESP', '1771 ESP', '402 ESP', '4 ESP', '1931 ESP', '3487 ESP', '259 ESP', '1235 ESP', '2065 ESP', '3034 ESP', '3152 ESP', '2290 ESP', '14 ESP', '3274 ESP', '3776 ESP', '2287 ESP', '338 ESP', '2044 ESP', '3034 ESP', '3034 ESP', '3656 ESP', '4076 ESP', '379 ESP', '3487 ESP']
  - P@50: 0.06, R@50: 0.6, MRR: 0.027777777777777776

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['1680 ESP', '4081 ESP', '1848 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '2335 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '3166 ESP', '4081 ESP', '4081 ESP', '1794 ESP', '1325 ESP', '1796 ESP', '2576 ESP', '1680 ESP', '3166 ESP', '4081 ESP', '4081 ESP', '1680 ESP', '2260 ESP', '4081 ESP', '571 ESP', '1680 ESP', '2335 ESP', '2335 ESP', '1794 ESP', '3392 ESP', '1646 ESP', '3331 ESP', '2335 ESP', '4081 ESP', '571 ESP', '2574 ESP', '2335 ESP', '1195 ESP', '3166 ESP', '2335 ESP', '3166 ESP', '1850 ESP', '1680 ESP', '2335 ESP', '1680 ESP', '1848 ESP', '4081 ESP', '3166 ESP', '1314 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.038461538461538464

