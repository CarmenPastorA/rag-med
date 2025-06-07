# Primary FAISS Evaluation - primary_faiss_k50

## Configuration
- Doc top-N: 50
- Device: cuda

## Global Results

- **mean_precision@k**: 0.0918
- **mean_normalized_precision@k**: 0.5799
- **mean_recall@k**: 0.5753
- **mean_hit@k**: 0.8521
- **mean_mrr**: 0.407
- **mean_n_relevant_retrieved**: 4.59
- **questions_evaluated**: 2630
- **doc_top_n**: 50
- **device**: cuda

---
## Sample Questions (first 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['4076 ESP', '3656 ESP', '3768 ESP', '2157 ESP', '3776 ESP', '14 ESP', '3152 ESP', '1053 ESP', '810 ESP', '821 ESP', '2852 ESP', '2531 ESP', '2409 ESP', '1917 ESP', '253 ESP', '3487 ESP', '3459 ESP', '2942 ESP', '980 ESP', '4240 ESP', '2665 ESP', '2699 ESP', '3034 ESP', '2575 ESP', '2277 ESP', '3759 ESP', '2630 ESP', '854 ESP', '1187 ESP', '3145 ESP', '58 IP', '61 ESP', '2657 ESP', '1252 ESP', '2289 ESP', '3175 ESP', '3839 ESP', '2987 ESP', '3503 ESP', '4023 ESP', '2138 ESP', '32 ESP', '971 ESP', '3414 ESP', '2631 ESP', '2784 ESP', '3591 ESP', '4215 ESP', '1169 ESP', '1251 ESP']
  - P@50: 0.14, R@50: 0.875, MRR: 1.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['4332 ESP', '4333 ESP', '4008 ESP', '4009 ESP', '3393 ESP', '4334 ESP', '3395 ESP', '638 ESP', '4013 ESP', '4012 ESP', '552 ESP', '1815 ESP', '397 ESP', '3002 ESP', '4077 ESP', '4010 ESP', '537 ESP', '4011 ESP', '1816 ESP', '536 ESP', '2222 ESP', '1580 ESP', '641 ESP', '637 ESP', '1663 ESP', '2780 ESP', '2437 ESP', '4174 ESP', '690 ESP', '4239 ESP', '2430 ESP', '4355 ESP', '2438 ESP', '307 ESP', '1664 ESP', '4179 ESP', '2223 ESP', '3068 ESP', '3627 ESP', '4326 ESP', '4233 ESP', '3764 ESP', '2428 ESP', '1435 ESP', '3804 ESP', '3868 ESP', '4161 ESP', '3095 ESP', '4014 ESP', '3189 ESP']
  - P@50: 0.06, R@50: 0.42857142857142855, MRR: 1.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2921 ESP', '4174 ESP', '1071 ESP', '816 ESP', '2994 ESP', '2519 ESP', '3825 ESP', '2466 ESP', '3092 ESP', '4401 ESP', '385 ESP', '4309 ESP', '2223 ESP', '1580 ESP', '3068 ESP', '4308 ESP', '4179 ESP', '2222 ESP', '3095 ESP', '1892 ESP', '4241 ESP', '4136 ESP', '4202 ESP', '2512 ESP', '2393 ESP', '2437 ESP', '2431 ESP', '3399 ESP', '3083 ESP', '2438 ESP', '2900 ESP', '2001 ESP', '3002 ESP', '4356 ESP', '2040 ESP', '1815 ESP', '3069 ESP', '2208 ESP', '2934 ESP', '4008 ESP', '4030 ESP', '2203 ESP', '2448 ESP', '1816 ESP', '4383 ESP', '3082 ESP', '4355 ESP', '3582 ESP', '4330 ESP', '3765 ESP']
  - P@50: 0.04, R@50: 0.6666666666666666, MRR: 0.1

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['2228 ESP', '3656 ESP', '971 ESP', '4076 ESP', '2157 ESP', '2029 ESP', '3152 ESP', '3768 ESP', '3003 ESP', '3346 ESP', '3034 ESP', '253 ESP', '2288 ESP', '3204 ESP', '2048 ESP', '3012 ESP', '1931 ESP', '61 ESP', '4056 ESP', '3776 ESP', '4240 ESP', '14 ESP', '431 ESP', '2789 ESP', '2279 ESP', '4097 ESP', '308 ESP', '903 ESP', '251 ESP', '3115 ESP', '3458 ESP', '130 ESP', '3128 ESP', '1771 ESP', '519 ESP', '3487 ESP', '58 IP', '1442 ESP', '3569 ESP', '3686 ESP', '3505 ESP', '3274 ESP', '854 ESP', '1279 ESP', '3503 ESP', '2492 ESP', '2338 ESP', '32 ESP', '162 ESP', '1353 ESP']
  - P@50: 0.04, R@50: 0.4, MRR: 0.09090909090909091

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['821 ESP', '571 ESP', '570 ESP', '402 ESP', '2223 ESP', '1646 ESP', '2222 ESP', '1796 ESP', '2288 ESP', '1680 ESP', '3592 ESP', '1855 ESP', '4379 ESP', '2335 ESP', '1794 ESP', '2280 ESP', '2221 ESP', '349 ESP', '251 ESP', '3166 ESP', '438 ESP', '1922 ESP', '3071 ESP', '162 ESP', '1663 ESP', '2574 ESP', '4081 ESP', '259 ESP', '3743 ESP', '1879 ESP', '1793 ESP', '3588 ESP', '841 ESP', '3346 ESP', '966 ESP', '2576 ESP', '2220 ESP', '2773 ESP', '439 ESP', '2260 ESP', '1019 ESP', '1662 ESP', '2437 ESP', '811 ESP', '3012 ESP', '519 ESP', '1848 ESP', '4136 ESP', '1174 ESP', '1541 ESP']
  - P@50: 0.06, R@50: 1.0, MRR: 1.0

