import json
from collections import Counter
from pathlib import Path

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
counts = Counter(c["file"] for c in chunks)
for file, n in sorted(counts.items()):
    print(f"{n:>4}  {file}")
print()
print("Total:", len(chunks), " Files with chunks:", len(counts))