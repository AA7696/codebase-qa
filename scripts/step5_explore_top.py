import ast
from pathlib import Path

repo = Path("data/repos/requests")
skip = {".git", ".venv", "__pycache__", "tests"}

hooks = repo / "src/requests/hooks.py"
tree = ast.parse(hooks.read_text(encoding="utf-8"))
print("Top level of hooks.py:")
for node in tree.body:
    print(f"  {type(node).__name__:<12} lines {node.lineno}-{node.end_lineno}")

total = 0
for path in sorted(repo.rglob("*.py")):
    if any(part in skip for part in path.relative_to(repo).parts):
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        continue
    total += sum(isinstance(n, (ast.Assign, ast.AnnAssign)) for n in tree.body)

print()
print("Top-level assignments across the repo (in no chunk today):", total)