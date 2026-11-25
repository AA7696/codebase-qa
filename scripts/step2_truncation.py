import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
model = SentenceTransformer("all-MiniLM-L6-v2")

for chunk in chunks:
    if chunk["name"] == "Session.__init__":
        token_ids = model.tokenizer.encode(chunk["code"], add_special_tokens=False)
        print("Total tokens in this chunk:", len(token_ids))

        seen_text = model.tokenizer.decode(token_ids[:256])
        print("Model saw the word 'cookie':", "cookie" in seen_text)
        print()
        print("The last part the model saw:")
        print(seen_text[-300:])