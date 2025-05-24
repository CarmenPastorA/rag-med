#grouping_utils.y
import os
import json
from typing import List, Dict, Any
from collections import Counter

# === Paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
print(f"PROJECT ROOT: {PROJECT_ROOT}")

def load_master_json(path: str) -> Dict[str, Any]:
    """
    Load structured medication metadata from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def filter_documents(master: Dict[str, Any], criteria: Dict[str, Any]) -> List[str]:
    """
    Filter all document IDs in the master dataset that match the given criteria.
    Fields in the criteria dictionary may be set to None to disable filtering by that attribute.
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

def extract_existing_groupings(master: Dict[str, Any], min_docs: int = 3) -> List[Dict[str, Any]]:
    """
    Extract all real (species, indication, route, antibiotic category) groupings that occur in the dataset,
    and filter them by minimum number of matching documents.
    Returns a list of dictionaries with grouping details and document count.
    """
    grouping_counter = Counter()

    for doc_id, entry in master.items():
        species = entry.get("especies", [])
        routes = entry.get("ruta_administracion", [])
        indications_dict = entry.get("indicaciones", {})
        categories = set()

        for ab in entry.get("antibiotico", []):
            if isinstance(ab, dict):
                cat = ab.get("Categoría")
                if cat:
                    categories.add(cat)

        for sp in species:
            indications = indications_dict.get(sp, []) or [None]
            routes_iter = routes or [None]
            categories_iter = categories or [None]

            for ind in indications:
                for route in routes_iter:
                    for cat in categories_iter:
                        grouping_counter[(sp, ind, route, cat)] += 1

    # Filter groupings by min_docs
    filtered = []
    for (sp, ind, route, cat), count in grouping_counter.items():
        if count >= min_docs:
            filtered.append({
                "especie": sp,
                "indicacion": ind,
                "via": route,
                "categoria_antibiotico": cat,
                "descripcion": " para " + ", ".join(
                    f"{k} del antibiótico: {v}" if k == "categoría" and v else f"{k}: {v}"
                    for k, v in {
                        "especie": sp,
                        "indicación": ind,
                        "vía": route,
                        "categoría": cat
                    }.items() if v
                ),
                "num_docs": count
            })

    return filtered
