"""Minimal example of querying a vLLM server using the OpenAI client."""

from openai import OpenAI

# Connect to a locally running vLLM server using the OpenAI compatible API.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # authentication is not required for local usage
)

# Send a single question to the model and print the response.
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {
            "role": "user",
            "content": (
                "¿Qué antibióticos se pueden usar en terneros con neumonía?"
            ),
        }
    ],
    temperature=0.2,
)

print("\n🧠 Model response:")
print(response.choices[0].message.content.strip())
