import ast
from pathlib import Path

repo = Path("data/repos/requests")
skip = {".git", ".venv", "__pycache__", "tests"}

found = []
for path in sorted(repo.rglob("*.py")):
    rel = path.relative_to(repo)
    if any(part in skip for part in rel.parts):
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        continue
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        name = target.id if isinstance(target, ast.Name) else "(other)"
        size = node.end_lineno - node.lineno + 1
        found.append((size, rel.as_posix(), name, node.lineno, node.end_lineno))

print("Total:", len(found))
print("One or two lines long:", sum(1 for f in found if f[0] <= 2))
print()
print("15 longest:")
for size, file, name, start, end in sorted(found, reverse=True)[:15]:
    print(f"  {size:>4} lines  {file}  {name}  ({start}-{end})")
print()
print("In hooks.py:")
for size, file, name, start, end in found:
    if file.endswith("hooks.py"):
        print(f"  {name}  ({start}-{end})")