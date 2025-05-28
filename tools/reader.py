# reader.py

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # dummy key
)

def generate_answer_with_vllm(question: str, context: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
    messages = [
        {"role": "system", "content": "Eres un asistente experto en medicamentos veterinarios. Responde siempre en español. Responde de forma clara y basada en la información proporcionada."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
