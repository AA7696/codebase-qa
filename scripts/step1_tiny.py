import ast
from pathlib import Path

file_path = Path("data/repos/requests/src/requests/sessions.py")
source = file_path.read_text(encoding="utf-8")

lines = source.splitlines()
tree = ast.parse(source)

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "merge_setting":
        start = node.lineno
        end = node.end_lineno
        print("Starts at line:", start)
        print("Ends at line:", end)
        print("---- the code ----")
        piece = lines[start - 1 : end]
        print("\n".join(piece))