import ast
from pathlib import Path

file_path = Path("data/repos/requests/src/requests/sessions.py")
source = file_path.read_text(encoding="utf-8")

lines = source.splitlines()
tree = ast.parse(source)

chunks = []


def make_chunk(node, name):
    return {
        "file": "src/requests/sessions.py",
        "name": name,
        "start": node.lineno,
        "end": node.end_lineno,
        "code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
    }


def is_stub(node):
    # A stub is a function whose whole body is just "..."
    if len(node.body) != 1:
        return False
    only_line = node.body[0]
    return (
        isinstance(only_line, ast.Expr)
        and isinstance(only_line.value, ast.Constant)
        and only_line.value.value is Ellipsis
    )


for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        chunks.append(make_chunk(node, node.name))

    elif isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and not is_stub(child):
                chunks.append(make_chunk(child, node.name + "." + child.name))

print("Number of chunks:", len(chunks))
for chunk in chunks:
    print(chunk["name"], chunk["start"], chunk["end"])

print()
print("Biggest chunks:")
by_size = sorted(chunks, key=lambda c: c["end"] - c["start"] + 1, reverse=True)
for chunk in by_size[:5]:
    print(chunk["name"], chunk["end"] - chunk["start"] + 1, "lines")

print()
print("Smallest chunks:")
for chunk in by_size[-5:]:
    print(chunk["name"], chunk["end"] - chunk["start"] + 1, "lines")