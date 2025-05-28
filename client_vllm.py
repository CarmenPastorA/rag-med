from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # no se valida
)

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {"role": "user", "content": "¿Qué antibióticos se pueden usar en terneros con neumonía?"}
    ],
    temperature=0.2
)

print("\n🧠 Respuesta del modelo:")
print(response.choices[0].message.content.strip())
