import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
print("Number of chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model reads at most:", model.max_seq_length, "tokens")

texts = [chunk["code"] for chunk in chunks]

too_long = 0
for text in texts:
    if len(model.tokenizer.encode(text)) > model.max_seq_length:
        too_long += 1
print("Chunks longer than that limit:", too_long)

embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

np.save("data/embeddings.npy", embeddings)
print("Saved to data/embeddings.npy")