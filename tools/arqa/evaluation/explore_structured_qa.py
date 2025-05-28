import os
import json
from collections import defaultdict
import pandas as pd

# === Ruta a tu dataset ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATASET_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")

# === Contenedores ===
grouped_questions = defaultdict(list)

# === Cargar y clasificar preguntas ===
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        qid = example.get("id")
        question = example.get("question")
        n_relevant = len(example.get("relevant_doc_ids", []))

        # Clasificación por rango
        if n_relevant <= 5:
            group = "3–5"
        elif n_relevant <= 10:
            group = "6–10"
        elif n_relevant <= 50:
            group = "11–50"
        else:
            group = ">50"

        grouped_questions[group].append({
            "rango_relevantes": group,
            "id_pregunta": qid,
            "n_relevantes": n_relevant,
            "pregunta": question
        })

# === Convertir a DataFrame y guardar/visualizar ===
df_grouped = pd.DataFrame(
    [item for group in grouped_questions.values() for item in group]
)

# Guardar CSV si lo necesitas
df_grouped.to_csv("preguntas_por_rango_relevantes.csv", index=False, encoding="utf-8")

# Muestra ejemplo
print(df_grouped.groupby("rango_relevantes").size())
print("\nEjemplo de preguntas del grupo '>50':")
print(df_grouped[df_grouped["rango_relevantes"] == ">50"].head())
