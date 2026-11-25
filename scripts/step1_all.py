import ast
from pathlib import Path

file_path = Path("data/repos/requests/src/requests/sessions.py")
source = file_path.read_text(encoding="utf-8")

lines = source.splitlines()
tree = ast.parse(source)

chunks = []

for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        chunk = {
            "file": "src/requests/sessions.py",
            "name": node.name,
            "start": node.lineno,
            "end": node.end_lineno,
            "code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
        }
        chunks.append(chunk)

print("Number of chunks:", len(chunks))
for chunk in chunks:
    print(chunk["name"], chunk["start"], chunk["end"])