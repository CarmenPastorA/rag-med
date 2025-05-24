import os
import json
import time
import random
import hashlib
import datetime
import torch
from openai import OpenAI
from grouping_utils import load_master_json, extract_existing_groupings, filter_documents

# === Constants ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# === Parameters ===
MODEL_NAME = "mistral"
MIN_DOCS = 3
TEST_MODE = False  # Set to False for full generation

# === Paths ===
INPUT_FILE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/json_data/master_merge_for_llm.json")
MODEL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "data/info_models/vllm.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"tools/arqa/evaluation/generated_datasets/structured_{MODEL_NAME}_min{MIN_DOCS}.jsonl")
DEBUG_GROUPINGS_PATH = os.path.join(PROJECT_ROOT, f"tools/arqa/evaluation/generated_datasets/groupings_debug_{MODEL_NAME}_min{MIN_DOCS}.jsonl")

# === Prompts used ===
SYSTEM_PROMPT = (
    "Eres un generador automático de preguntas clínicas en español para profesionales veterinarios. "
    "Tu tarea es redactar exclusivamente una pregunta clara y útil en español, basada en la necesidad clínica indicada. "
    "No debes incluir explicaciones, respuestas ni traducciones. "
    "Utiliza únicamente terminología clínica habitual en medicina veterinaria. "
    "Evita tecnicismos innecesarios o poco frecuentes. "
    "Evita el uso de palabras mal traducidas del inglés o términos inusuales. "
    "Escribe con corrección gramatical y ortográfica en español clínico. "
    "Ejemplo de formato esperado: "
    "¿Qué medicamentos están indicados para tratar la infección respiratoria causada por Pasteurella multocida en terneros por vía subcutánea?"
)

USER_PROMPT_TEMPLATE = (
    "La necesidad clínica es la siguiente: {descripcion}.\n"
    "Redacta solo una pregunta en español que refleje esta necesidad."
)

# === Load data ===
master = load_master_json(INPUT_FILE)
all_valid_groupings = extract_existing_groupings(master, min_docs=MIN_DOCS)
print(f"Total valid groupings found (MIN_DOCS ≥ {MIN_DOCS}): {len(all_valid_groupings)}")

# Save groupings for inspection
with open(DEBUG_GROUPINGS_PATH, "w", encoding="utf-8") as dbg_f:
    for g in all_valid_groupings:
        dbg_f.write(json.dumps(g, ensure_ascii=False) + "\n")

# Select subset if in test mode
if TEST_MODE:
    NUM_GROUPINGS = 10
    groupings = random.sample(all_valid_groupings, k=min(NUM_GROUPINGS, len(all_valid_groupings)))
    print(f"Test mode: generating {len(groupings)} questions.")
else:
    groupings = all_valid_groupings

# Load model config
with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
    model_config = json.load(f)[MODEL_NAME]

client = OpenAI(base_url=model_config["endpoint"], api_key="not-needed")

# === Main loop ===
start_time = time.time()
total_written = 0

with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
    for i, grouping in enumerate(groupings):
        doc_ids = filter_documents(master, grouping)
        if len(doc_ids) < MIN_DOCS:
            continue

        prompt = USER_PROMPT_TEMPLATE.format(descripcion=grouping["descripcion"])

        try:
            response = client.chat.completions.create(
                model=model_config["model_name"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config.get("temperature", 0.3)
            )

            question = response.choices[0].message.content.strip()

            out_f.write(json.dumps({
                "id": f"q_{i:04d}",
                "question": question,
                "relevant_doc_ids": doc_ids,
                "criteria": grouping,
                "model": MODEL_NAME
            }, ensure_ascii=False) + "\n")

            total_written += 1

            if i % 100 == 0:
                out_f.flush()
                print(f"{i} questions processed...")

        except Exception as e:
            print(f"[Error] Failed on grouping {i}: {grouping['descripcion']} → {e}")

total_time = time.time() - start_time

# === Metadata and prompt logging ===

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(device_id),
        "device_id": device_id,
        "total_memory_GB": round(props.total_memory / (1024 ** 3), 2),
        "memory_allocated_GB": round(torch.cuda.memory_allocated(device_id) / (1024 ** 3), 2),
        "memory_reserved_GB": round(torch.cuda.memory_reserved(device_id) / (1024 ** 3), 2),
        "cuda_version": torch.version.cuda
    }

def compute_file_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

METADATA_PATH = OUTPUT_PATH.replace(".jsonl", "_metadata.json")
PROMPT_PATH = OUTPUT_PATH.replace(".jsonl", "_prompts.txt")

metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "total_runtime_seconds": round(total_time, 2),
    "model": MODEL_NAME,
    "model_endpoint": model_config["endpoint"],
    "temperature": model_config.get("temperature", 0.3),
    "min_docs": MIN_DOCS,
    "total_groupings": len(all_valid_groupings),
    "total_questions_written": total_written,
    "output_file": OUTPUT_PATH,
    "output_file_sha256": compute_file_sha256(OUTPUT_PATH),
    "gpu_info": get_gpu_info()
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

with open(PROMPT_PATH, "w", encoding="utf-8") as f:
    f.write("=== SYSTEM MESSAGE ===\n")
    f.write(SYSTEM_PROMPT + "\n\n")
    f.write("=== USER PROMPT TEMPLATE ===\n")
    f.write(USER_PROMPT_TEMPLATE + "\n")
    f.write("=== DESCRIPCIÓN TEMPLATE ===\n")
    f.write(
        'para especie: {especie}, indicación: {indicacion}, vía: {via}, '
        'categoría del antibiótico: {categoria_antibiotico}\n'
    )

print(f"Finished. Total questions written: {total_written}")
print(f"Metadata saved to: {METADATA_PATH}")
print(f"Prompt definitions saved to: {PROMPT_PATH}")
