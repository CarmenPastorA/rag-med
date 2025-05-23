import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import random
from openai import OpenAI
from collections import defaultdict

# === Path setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)
print(f"PROJECT ROOT: {PROJECT_ROOT}")

from shared import dunder_info
dunder_info.inject_dunder(__name__)

# === Config paths ===
INPUT_FILE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/json_data/master_merge_for_llm.json")
MODEL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "data/info_models/vllm.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GroupedQAGenerator:
    def __init__(self, model_config: Dict, model_name: str, group_by: str, max_docs_per_prompt: int = 10):
        self.input_path = Path(INPUT_FILE)
        self.group_by = group_by
        self.model_name = model_name
        self.model_config = model_config[model_name]
        self.max_docs_per_prompt = max_docs_per_prompt
        self.output_path = Path(OUTPUT_DIR) / f"synthetic_{group_by}_{model_name}_{max_docs_per_prompt}.jsonl"
        self.log_path = self.output_path.with_suffix(".md")

        self.client = OpenAI(
            base_url=self.model_config["endpoint"],
            api_key="not-needed"
        )

        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.grouped_data = self._group_documents()

    def _group_documents(self) -> Dict[str, List[str]]:
        def extract_nested_value(entry: Dict, path: str):
            keys = path.split(".")
            val = entry
            for key in keys:
                if isinstance(val, list):
                    val = [v.get(key) for v in val if isinstance(v, dict) and key in v]
                elif isinstance(val, dict):
                    val = val.get(key)
                else:
                    return None
            return val

        grouped = defaultdict(list)
        for doc_id, entry in self.data.items():
            val = extract_nested_value(entry, self.group_by)
            if val is None:
                continue
            if isinstance(val, list):
                for v in val:
                    grouped[str(v)].append(doc_id)
            else:
                grouped[str(val)].append(doc_id)
        return grouped

    def build_context(self, doc_ids: List[str]) -> str:
        fragments = []
        for doc_id in doc_ids:
            e = self.data[doc_id]
            especies = e.get("especies", [])
            indicaciones = e.get("indicaciones", {})
            especie_indicaciones = [
                f"  - {esp}: {', '.join(indicaciones.get(esp, []))}"
                for esp in especies if esp in indicaciones and indicaciones.get(esp)
            ]
            indicaciones_str = "\n".join(especie_indicaciones) if especie_indicaciones else "  - No especificadas"

            antibiotico = e.get("antibiotico", "No")
            if isinstance(antibiotico, str) and antibiotico.strip().lower() == "no":
                antibiotico_str = "Este medicamento no contiene antibióticos."
            elif isinstance(antibiotico, list):
                lines = []
                for a in antibiotico:
                    if isinstance(a, dict):
                        categoria = a.get("Categoría", "No especificada")
                        familia = a.get("Familia", "No especificada")
                        principio = a.get("Principio activo", "No especificada")
                        lines.append(
                            f"  - Principio activo: {principio}\n"
                            f"    Familia: {familia}\n"
                            f"    Categoría: {categoria}"
                        )
                antibiotico_str = "Antibióticos:\n" + "\n".join(lines) if lines else "No especificado"
            else:
                antibiotico_str = "Formato desconocido para el campo 'antibiotico'"

            frag = (
                f"Nombre: {e['nombre']}\n"
                f"Especies: {', '.join(especies)}\n"
                f"Indicaciones por especie:\n{indicaciones_str}\n"
                f"Antibiótico: {antibiotico_str}\n"
                f"Vías: {', '.join(e.get('ruta_administracion', []))}"
            )
            fragments.append(frag)
        return "\n\n".join(fragments)

    def generate_prompt(self, group_name: str, doc_ids: List[str]) -> str:
        context = self.build_context(doc_ids)
        return (
            f"Eres un experto en farmacología veterinaria. A continuación se presenta información sobre medicamentos "
            f"clasificados por {self.group_by.lower()}: '{group_name}'.\n"
            "Genera una pregunta útil y realista que se pueda responder exclusivamente con la información proporcionada. "
            "Debe combinar conceptos como especie, indicación, vía de administración o grupo antibiótico. "
            "Responde siempre en español. Devuelve solo la pregunta, sin explicación:\n\n"
            f"{context}\n\nPregunta:"
        )

    def query_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_config["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en farmacología veterinaria especializado en redacción de preguntas. "
                        "Tu tarea es generar únicamente preguntas generales y útiles que puedan ser respondidas con el contexto proporcionado. "
                        "No incluyas respuestas, explicaciones ni ejemplos."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.model_config.get("temperature", 0.3)
        )
        return response.choices[0].message.content.strip()

    def generate_dataset(self, questions_per_group: int = 3) -> None:
        # Crear archivo markdown de log
        with open(self.log_path, "w", encoding="utf-8") as log_f, open(self.output_path, "w", encoding="utf-8") as out_f:
            log_f.write(f"# Registro de generación - Modelo: `{self.model_name}`\n\n")
    
            for group_name, doc_ids in self.grouped_data.items():
                if len(doc_ids) < 2:
                    continue
    
                for i in range(questions_per_group):
                    sampled_doc_ids = random.sample(doc_ids, min(len(doc_ids), self.max_docs_per_prompt))
                    prompt = self.generate_prompt(group_name, sampled_doc_ids)
    
                    try:
                        question = self.query_llm(prompt)
    
                        # Guardar pregunta en JSONL
                        out_f.write(json.dumps({
                            "question": question,
                            "relevant_doc_ids": sampled_doc_ids
                            #"group": group_name,
                            #"group_by": self.group_by,
                            #"model": self.model_name
                        }, ensure_ascii=False) + "\n")
    
                        # Guardar en log markdown
                        log_f.write(f"## Grupo: {group_name}\n")
                        log_f.write(f"**Prompt:**\n\n```text\n{prompt}\n```\n")
                        log_f.write(f"**Pregunta generada:**\n\n> {question}\n\n---\n")
    
                    except Exception as e:
                        print(f"Error generating question for group {group_name}: {e}")


if __name__ == "__main__":
    with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
        model_key = "mistral"

    #for model_key in ["mistral", "qwen2.5"]:
    generator = GroupedQAGenerator(
        model_config=config,
        model_name=model_key,
        group_by="especies",
        max_docs_per_prompt=10
    )
    generator.generate_dataset(questions_per_group=1)


