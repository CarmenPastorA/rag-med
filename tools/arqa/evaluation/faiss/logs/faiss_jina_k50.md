# FAISS Evaluation Report - faiss_eval_jina_2025-06-20_k50

## Configuration
- `Note: Improved FAISS index adding 'Nombre Medicamento: ' at the beginning of each chunk`
- `Top-k`: 50
- `Embedding model`: `jinaai/jina-embeddings-v3`
- `Device`: `cuda`
- `Date`: 2025-06-20 10:25

## Global Results

- **Mean precision@k**: 0.0582
- **Mean normalized precision@k**: 0.3961
- **Mean recall@k**: 0.3925
- **Mean hit@k**: 0.5262
- **Mean mrr**: 0.1652
- **Mean n relevant retrieved**: 2.91
- **Questions evaluated**: 2630
- **K**: 50
- **Embedding model**: jinaai/jina-embeddings-v3
- **Device**: cuda
- **Run name**: faiss_eval_jina_2025-06-20_k50

---
## Sample Questions (Top 5)

- **Question:** ¿Qué antibióticos de categoría B están indicados para tratar la septicemia por Escherichia coli en ovinos a través de la vía intramuscular?
  - Relevant doc IDs: ['3768 ESP', '4076 ESP', '14 ESP', '13 ESP', '3776 ESP', '2630 ESP', '2157 ESP', '3656 ESP']
  - Retrieved doc IDs: ['3175 ESP', '3175 ESP', '3175 ESP', '2835 ESP', '3175 ESP', '4023 ESP', '3152 ESP', '1169 ESP', '3024 ESP', '1252 ESP', '2942 ESP', '2835 ESP', '2835 ESP', '2835 ESP', '2835 ESP', '3152 ESP', '2835 ESP', '2835 ESP', '3152 ESP', '2835 ESP', '2835 ESP', '2987 ESP', '2835 ESP', '4104 ESP', '1169 ESP', '1738 ESP', '2752 ESP', '3533 ESP', '3175 ESP', '4399 ESP', '2835 ESP', '3382 ESP', '1281 ESP', '4201 ESP', '2835 ESP', '2835 ESP', '2856 ESP', '2856 ESP', '2835 ESP', '1169 ESP', '2754 ESP', '4399 ESP', '1293 ESP', '2990 ESP', '4023 ESP', '3425 ESP', '914 ESP', '2835 ESP', '2835 ESP', '2610 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

- **Question:** ¿Qué medicamento oral de categoría D se recomienda para tratar la infección urogenital en perros causada por microorganismos sensibles, administrándose por vía oral?
  - Relevant doc IDs: ['1822 ESP', '848 ESP', '840 ESP', '842 ESP', '4333 ESP', '4332 ESP', '4334 ESP']
  - Retrieved doc IDs: ['3581 ESP', '3126 ESP', '3159 ESP', '30 IP', '3582 ESP', '1891 ESP', '1380 ESP', '3126 ESP', '1656 ESP', '3124 ESP', '1656 ESP', '3581 ESP', '3583 ESP', '3582 ESP', '3583 ESP', '3627 ESP', '3178 ESP', '1891 ESP', '1374 ESP', '3772 ESP', '1418 ESP', '3806 ESP', '3587 ESP', '3126 ESP', '3371 ESP', '3008 ESP', '3126 ESP', '3066 ESP', '3124 ESP', '3516 ESP', '1373 ESP', '3746 ESP', '3124 ESP', '3159 ESP', '3978 ESP', '820 ESP', '4334 ESP', '3124 ESP', '4093 ESP', '662 ESP', '1853 ESP', '1491 ESP', '3460 ESP', '29 IP', '3783 ESP', '3126 ESP', '2442 ESP', '3587 ESP', '3773 ESP', '3787 ESP']
  - P@50: 0.02, R@50: 0.14285714285714285, MRR: 0.02702702702702703

- **Question:** ¿Qué medicamentos recomiendan para el tratamiento de las náuseas perioperatorias en perros a través de vía subcutánea?
  - Relevant doc IDs: ['EU/2/17/211/001', '4202 ESP', '4401 ESP']
  - Retrieved doc IDs: ['2001 ESP', '2466 ESP', '1986 ESP', '3700 ESP', '3733 ESP', '4241 ESP', '3510 ESP', '1892 ESP', '3930 ESP', '3930 ESP', '3453 ESP', '2413 ESP', '2264 ESP', '3931 ESP', '3844 ESP', '1986 ESP', '3642 ESP', '2431 ESP', '3931 ESP', '3930 ESP', '3510 ESP', '4030 ESP', '3931 ESP', '2346 ESP', '3510 ESP', '3930 ESP', '3510 ESP', '3928 ESP', '3929 ESP', '2262 ESP', '3778 ESP', '3120 ESP', '4202 ESP', '2001 ESP', '3930 ESP', '2466 ESP', '3931 ESP', '3120 ESP', '2176 ESP', '3932 ESP', '3929 ESP', '3932 ESP', '3700 ESP', '1624 ESP', '3932 ESP', '2180 ESP', '3929 ESP', '3120 ESP', '3931 ESP', '2263 ESP']
  - P@50: 0.02, R@50: 0.3333333333333333, MRR: 0.030303030303030304

- **Question:** ¿Qué antibióticos de la categoría B están indicados para tratar la infección gastrointestinal de Escherichia coli en terneros a través de vía intravenosa?
  - Relevant doc IDs: ['3034 ESP', '4239 ESP', '13 ESP', '944 ESP', '903 ESP']
  - Retrieved doc IDs: ['1298 ESP', '3382 ESP', '3175 ESP', '3175 ESP', '3175 ESP', '3175 ESP', '4023 ESP', '3382 ESP', '3629 ESP', '3152 ESP', '3569 ESP', '2835 ESP', '1169 ESP', '1915 ESP', '1620 ESP', '2228 ESP', '2942 ESP', '3175 ESP', '1738 ESP', '2048 ESP', '1860 ESP', '3937 ESP', '2044 ESP', '3128 ESP', '4 ESP', '3879 ESP', '2230 ESP', '3693 ESP', '4201 ESP', '2752 ESP', '2029 ESP', '3692 ESP', '3577 ESP', '4399 ESP', '3450 ESP', '1169 ESP', '2835 ESP', '2835 ESP', '2230 ESP', '2835 ESP', '2770 ESP', '2545 ESP', '2289 ESP', '3961 ESP', '3487 ESP', '3999 ESP', '3950 ESP', '3175 ESP', '2835 ESP', '2835 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

- **Question:** ¿Qué antibióticos están indicados para tratar las infecciones causadas por Salmonella spp. en Pavos a través de la vía oral, pertenecientes a la categoría B (uso restringido)?
  - Relevant doc IDs: ['821 ESP', '570 ESP', '571 ESP']
  - Retrieved doc IDs: ['2335 ESP', '3166 ESP', '1680 ESP', '2335 ESP', '2335 ESP', '1680 ESP', '1680 ESP', '3166 ESP', '2335 ESP', '2335 ESP', '1646 ESP', '2335 ESP', '1855 ESP', '3166 ESP', '1680 ESP', '4081 ESP', '4081 ESP', '4027 ESP', '2335 ESP', '4081 ESP', '1680 ESP', '4081 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '1646 ESP', '2335 ESP', '4081 ESP', '4027 ESP', '1680 ESP', '1680 ESP', '1680 ESP', '3166 ESP', '4081 ESP', '3166 ESP', '3693 ESP', '4081 ESP', '1680 ESP', '2335 ESP', '2335 ESP', '4081 ESP', '3392 ESP', '4081 ESP', '1680 ESP', '1680 ESP', '3166 ESP', '4081 ESP', '1646 ESP', '2335 ESP', '1680 ESP']
  - P@50: 0.0, R@50: 0.0, MRR: 0.0

