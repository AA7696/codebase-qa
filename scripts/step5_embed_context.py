import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [f"File: {c['file']}\nSymbol: {c['name']}\n{c['code']}" for c in chunks]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
np.save("data/embeddings_context.npy", embeddings)
print("Saved", embeddings.shape)