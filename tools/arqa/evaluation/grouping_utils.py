# === Path setup ===
import os
import json
import random
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
print(f"PROJECT ROOT: {PROJECT_ROOT}")

# === Grouping Utilities ===
def load_master_json(path: str) -> Dict[str, Any]:
    """
    Load structured medication metadata from master_merge_for_llm.json.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_unique_values(master: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract unique values for species, indications, administration routes, antibiotic categories,
    and build valid (species, indication) pairs.
    """
    especies, indicaciones, vias, categorias = set(), set(), set(), set()
    especie_indicacion_pairs = set()

    for entry in master.values():
        especies.update(entry.get("especies", []))
        vias.update(entry.get("ruta_administracion", []))

        for especie, indic_list in entry.get("indicaciones", {}).items():
            indicaciones.update(indic_list)
            for indic in indic_list:
                especie_indicacion_pairs.add((especie, indic))

        ab = entry.get("antibiotico", [])
        if isinstance(ab, list):
            for a in ab:
                if isinstance(a, dict):
                    cat = a.get("Categoría")
                    if cat:
                        categorias.add(cat)

    return {
        "especie": especies,
        "indicacion": indicaciones,
        "via": vias,
        "categoria_antibiotico": categorias,
        "especie_indicacion_pairs": especie_indicacion_pairs
    }

def sample_groupings(possible_values: Dict[str, Any], n: int = 100) -> List[Dict[str, str]]:
    """
    Create n random valid groupings using combinations of available criteria.
    If any field is None, it won't be filtered.
    """
    all_groupings = []
    for _ in range(n):
        especie = random.choice(list(possible_values["especie"]))
        indicacion = random.choice(list(possible_values["indicacion"])) if possible_values["indicacion"] else None
        via = random.choice(list(possible_values["via"])) if possible_values["via"] else None
        categoria = random.choice(list(possible_values["categoria_antibiotico"])) if possible_values["categoria_antibiotico"] else None

        criterios = [f"especie: {especie}"]
        if indicacion: criterios.append(f"indicación: {indicacion}")
        if via: criterios.append(f"vía: {via}")
        if categoria: criterios.append(f"categoría: {categoria}")

        descripcion = " para " + ", ".join(criterios)

        grouping = {
            "especie": especie,
            "indicacion": indicacion,
            "via": via,
            "categoria_antibiotico": categoria,
            "descripcion": descripcion
        }
        all_groupings.append(grouping)
    return all_groupings

def filter_documents(master: Dict[str, Any], criteria: Dict[str, Any]) -> List[str]:
    """
    Filter all medication IDs that match the selected grouping criteria.
    Supports flexible filtering: pass None to skip a criterion.
    """
    result = []
    for doc_id, entry in master.items():
        if criteria["especie"] and criteria["especie"] not in entry.get("especies", []):
            continue

        if criteria["indicacion"]:
            indicaciones = entry.get("indicaciones", {}).get(criteria["especie"], [])
            if criteria["indicacion"] not in indicaciones:
                continue

        if criteria["via"]:
            if criteria["via"] not in entry.get("ruta_administracion", []):
                continue

        if criteria["categoria_antibiotico"]:
            ab = entry.get("antibiotico", [])
            if isinstance(ab, list):
                found = any(isinstance(a, dict) and a.get("Categoría") == criteria["categoria_antibiotico"] for a in ab)
                if not found:
                    continue
            else:
                continue

        result.append(doc_id)
    return result


