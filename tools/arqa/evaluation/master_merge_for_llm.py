"""
This script processes a the metadata JSON file (`master_merge.json`)
and normalizes it into a simplified format suitable for downstream use (e.g. QA generation or LLM input).

The output format has been designed based on iterations with the LLM (ChatGPT 4o PRO), passing the original json and asking
how I should structure it so it can generate Q&A pairs.

Example:
{
  "3934 ESP": {
    "nombre": "X",
    "antibiotico": "C - Usar con cautela",
    "active_substances": ["Amoxicilina"],
    "especies": ["Bovino", "Porcino"],
    "ruta_administracion": ["Intramuscular"],
    "indicaciones": {
      "Bovino": ["Neumonía bacteriana"],
      "Porcino": ["Infecciones digestivas"]
    }
  },
  ...
}
"""

import json
import os
import sys

# === Path setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # tools/arqa/evaluation
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))  # root directory
sys.path.append(PROJECT_ROOT)
print(f"PROJECT ROOT: {PROJECT_ROOT}")

# === Shared module loader ===
from shared import dunder_info
dunder_info.inject_dunder(__name__)  # injects shared variables

# === Input/Output paths ===
INPUT_FILE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/json_data/master_merge.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/json_data/master_merge_for_llm.json")

def normalize_medications(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized_data = {}

    # Expecting input to be a dict like {"3934 ESP": { ... }}
    if isinstance(raw_data, dict):
        for med_id, wrapper in raw_data.items():
            antibiotico = wrapper.get("antibiotico", [])
            if not antibiotico:
                antibiotico = "No"

            entry = wrapper.get("metadata", {})  # Extract 'metadata' block

            name = entry.get("nombre", "")
            principio_activo = [pa["nombre"] for pa in entry.get("principiosActivos", [])]
            ruta_administracion = [route["nombre"] for route in entry.get("viasAdministracion", [])]
            especies = [sp["nombre"] for sp in entry.get("especies", [])]

            # Group indications by species
            indicacion_por_especies = {}
            for ind in entry.get("indicaciones", []):
                sp = ind.get("especie", {}).get("nombre", "All")
                indicacion_por_especies.setdefault(sp, []).append(ind["nombre"])

            normalized_data[med_id] = {
                "nombre": name,
                "antibiotico": antibiotico,
                "active_substances": principio_activo,
                "especies": especies,
                "ruta_administracion": ruta_administracion,
                "indicaciones": indicacion_por_especies
            }
    else:
        raise ValueError("Expected input JSON to be a dictionary with 'metadata' fields inside.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"Normalized data saved to: {output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    normalize_medications(INPUT_FILE, OUTPUT_FILE)

