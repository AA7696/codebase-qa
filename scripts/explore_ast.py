import ast
from pathlib import Path

source = Path("data/repos/requests/src/requests/sessions.py").read_text(encoding="utf-8")
tree = ast.parse(source)

for node in tree.body:
    print(type(node).__name__, getattr(node, "name", ""), getattr(node, "lineno", ""))


for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "Session":
        print("Session:", node.lineno, node.end_lineno)
        for child in node.body:
            print("  ", type(child).__name__, getattr(child, "name", ""), child.lineno, child.end_lineno)