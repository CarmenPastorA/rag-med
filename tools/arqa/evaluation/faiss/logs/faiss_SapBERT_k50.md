# FAISS Evaluation Report - faiss_eval_SapBERT_2025-06-19_k50

## Configuration
- `Note: Improved FAISS index adding 'Nombre Medicamento: ' at the beginning of each chunk`
- `Top-k`: 50
- `Embedding model`: `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`
- `Device`: `cuda`
- `Date`: 2025-06-19 16:57

## Global Results

- **Mean precision@k**: 0.0298
- **Mean normalized precision@k**: 0.1798
- **Mean recall@k**: 0.1775
- **Mean hit@k**: 0.3989
- **Mean mrr**: 0.1005
- **Mean n relevant retrieved**: 1.49
- **Questions evaluated**: 2630
- **K**: 50
- **Embedding model**: cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR
- **Device**: cuda
- **Run name**: faiss_eval_SapBERT_2025-06-19_k50

---
## Sample Questions (Top 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['313 ESP', '3152 ESP', '1177 ESP', '2059 ESP', '4082 ESP', '3832 ESP', '4377 ESP', '3951 ESP', '2025 ESP', '1195 ESP', '2079 ESP', '4101 ESP', '3567 ESP', '2582 ESP', '1885 ESP', '1720 ESP', '2566 ESP', '259 ESP', '2035 ESP', '2335 ESP', '1178 ESP', '1659 ESP', '1954 ESP', '1563 ESP', '3607 ESP', '1924 ESP', '3603 ESP', '1606 ESP', '1413 ESP', '1458 ESP', '4315 ESP', '3710 ESP', '1927 ESP', '2547 ESP', '3839 ESP', '2080 ESP', '4102 ESP', '4398 ESP', '2472 ESP', '3781 ESP', '3878 ESP', '2761 ESP', '3697 ESP', '273 ESP', '2256 ESP', '2892 ESP', '3792 ESP', '3592 ESP', '3781 ESP', '1252 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['2214 ESP', '2197 ESP', '1944 ESP', '3540 ESP', '3542 ESP', '3362 ESP', '3783 ESP', '1943 ESP', '3782 ESP', '3361 ESP', '2114 ESP', '4355 ESP', '2258 ESP', '4089 ESP', '2522 ESP', '3627 ESP', '4088 ESP', '2934 ESP', '3161 ESP', '2521 ESP', '4356 ESP', '4034 ESP', '4231 ESP', '2582 ESP', '4035 ESP', '4036 ESP', '3158 ESP', '2955 ESP', '431 ESP', '2523 ESP', '3094 ESP', '2215 ESP', '3786 ESP', '2032 ESP', '4328 ESP', '4087 ESP', '2520 ESP', '3158 ESP', '4179 ESP', '4244 ESP', '4033 ESP', '1505 ESP', '3283 ESP', '4327 ESP', '2956 ESP', '2957 ESP', '3787 ESP', '2174 ESP', '4326 ESP', '4407 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['51505 IP', '1610 ESP', '51506 IP', '792 ESP', '1183 ESP', '1892 ESP', '4401 ESP', '3772 ESP', '51507 IP', '3773 ESP', '3326 ESP', '1556 ESP', '2163 ESP', '3733 ESP', '51504 IP', '1986 ESP', '3599 ESP', '4062 ESP', '3361 ESP', '3362 ESP', '3327 ESP', '983 ESP', '4042 ESP', '2162 ESP', '52 IP', '3010 ESP', '1554 ESP', '1555 ESP', '53 IP', '3929 ESP', '3979 ESP', '3930 ESP', '2197 ESP', '54 IP', '3931 ESP', '4349 ESP', '3328 ESP', '2263 ESP', '4061 ESP', '208 ESP', '2431 ESP', '3540 ESP', '3329 ESP', '3069 ESP', '2881 ESP', '3932 ESP', '3849 ESP', '4063 ESP', '4062 ESP', '1191 ESP']
  - P@50: 0.02, R@50: 0.3333333333333333, MRR: 0.14285714285714285

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['4 ESP', '2044 ESP', '2256 ESP', '2566 ESP', '2582 ESP', '4101 ESP', '2025 ESP', '1298 ESP', '4102 ESP', '1177 ESP', '901 ESP', '1894 ESP', '3152 ESP', '431 ESP', '3832 ESP', '4082 ESP', '2789 ESP', '3691 ESP', '1339 ESP', '641 ESP', '306 ESP', '3279 ESP', '3878 ESP', '2042 ESP', '1458 ESP', '369 ESP', '3781 ESP', '2981 ESP', '2623 ESP', '3164 ESP', '3416 ESP', '3592 ESP', '3422 ESP', '4315 ESP', '3983 ESP', '2044 ESP', '2181 ESP', '3839 ESP', '3984 ESP', '3475 ESP', '690 ESP', '1235 ESP', '3759 ESP', '1887 ESP', '3904 ESP', '3331 ESP', '3458 ESP', '1620 ESP', '3116 ESP', '2940 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['4027 ESP', '4027 ESP', '1720 ESP', '1855 ESP', '431 ESP', '3802 ESP', '1680 ESP', '2275 ESP', '4101 ESP', '2940 ESP', '2566 ESP', '2242 ESP', '2508 ESP', '4102 ESP', '2335 ESP', '4082 ESP', '1112 ESP', '2244 ESP', '3697 ESP', '4198 ESP', '3731 ESP', '2361 ESP', '4199 ESP', '1693 ESP', '2940 ESP', '4377 ESP', '3839 ESP', '3607 ESP', '3802 ESP', '1907 ESP', '2623 ESP', '3386 ESP', '2953 ESP', '3902 ESP', '3717 ESP', '3416 ESP', '730 ESP', '3878 ESP', '3461 ESP', '2237 ESP', '3419 ESP', '2237 ESP', '4083 ESP', '2599 ESP', '3939 ESP', '4066 ESP', '2651 ESP', '2718 ESP', '2238 ESP', '1479 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

