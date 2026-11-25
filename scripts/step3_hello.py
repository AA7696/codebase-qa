import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from the environment
model = os.environ["OPENAI_MODEL"]

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "In one sentence, what is a Python decorator?"},
    ],
)

print("Model:", model)
print("Answer:", response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)