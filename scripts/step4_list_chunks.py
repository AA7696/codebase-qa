import json
from pathlib import Path

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
lines = [f"{c['file']}  |  {c['name']}  |  lines {c['start']}-{c['end']}" for c in chunks]
Path("data/chunk_names.txt").write_text("\n".join(lines), encoding="utf-8")
print("Wrote", len(lines), "names to data/chunk_names.txt")