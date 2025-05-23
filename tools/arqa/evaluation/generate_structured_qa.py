# === Path setup ===
import os
import json
import random
from pathlib import Path
from openai import OpenAI
from grouping_utils import load_master_json, extract_unique_values, sample_groupings, filter_documents

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# === Parameters ===
NUM_GROUPINGS = 100
MIN_DOCS = 2
MODEL_NAME = "mistral"  # Must match a key in llm_models_config.json

# === Config paths ===
INPUT_FILE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/json_data/master_merge_for_llm.json")
MODEL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "data/info_models/vllm.json")
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    f"tools/arqa/evaluation/generated_datasets/structured_{MODEL_NAME}_{NUM_GROUPINGS}_min{MIN_DOCS}.jsonl"
)

# === Load data ===
master = load_master_json(INPUT_FILE)
possible_values = extract_unique_values(master)
possible_values["indicacion"] = set()         # disables indication
possible_values["via"] = set()                # disables route of administration
groupings = sample_groupings(possible_values, NUM_GROUPINGS)

with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
    model_config = json.load(f)[MODEL_NAME]

client = OpenAI(
    base_url=model_config["endpoint"],
    api_key="not-needed"
)

# === Generate synthetic QA ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
    for grouping in groupings:
        doc_ids = filter_documents(master, grouping)
        print("\n=== AGRUPACIÓN ===")
        print(json.dumps(grouping, indent=2, ensure_ascii=False))
        print(f"# documentos coincidentes: {len(doc_ids)}")
        print("-" * 40)
        if len(doc_ids) < MIN_DOCS:
            continue

        continue

        prompt = (
            f"Eres un experto en farmacología veterinaria. Redacta una pregunta clara y general en español "
            f"basada en la siguiente necesidad clínica: {grouping['descripcion']}.\n"
            f"No incluyas nombres comerciales ni respuestas explícitas."
        )


        try:
            response = client.chat.completions.create(
                model=model_config["model_name"],
                messages=[
                    {
                        "role": "system",
                        "content": "Genera exclusivamente una pregunta en español, sin explicaciones ni respuestas. "
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=model_config.get("temperature", 0.3)
            )
            question = response.choices[0].message.content.strip()

            out_f.write(json.dumps({
                "question": question,
                "relevant_doc_ids": doc_ids,
                "criteria": grouping,
                "model": MODEL_NAME
            }, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[Error] {grouping} → {e}")

