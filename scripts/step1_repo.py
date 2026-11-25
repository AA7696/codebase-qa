import ast
import json
from pathlib import Path

repo_path = Path("data/repos/requests")
skip_folders = {".git", ".venv", "__pycache__", "tests"}


def is_stub(node):
    if len(node.body) != 1:
        return False
    only_line = node.body[0]
    return (
        isinstance(only_line, ast.Expr)
        and isinstance(only_line.value, ast.Constant)
        and only_line.value.value is Ellipsis
    )


def chunk_file(file_path):
    """Cut one file into chunks. Returns a list of dictionaries."""
    source = file_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    relative_name = file_path.relative_to(repo_path).as_posix()

    def first_line_of(node):
        # Start at the decorator line (like @property) if there is one
        if node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    def make_chunk(node, name):
        start = first_line_of(node)
        return {
            "file": relative_name,
            "name": name,
            "start": start,
            "end": node.end_lineno,
            "code": "\n".join(lines[start - 1 : node.end_lineno]),
        }

    def make_header_chunk(class_node, first_method):
        # The class line, its docstring and attributes, up to the first method
        start = first_line_of(class_node)
        if first_method is None:
            end = class_node.end_lineno
        else:
            end = first_line_of(first_method) - 1
        # Do not end on blank lines
        while end > start and lines[end - 1].strip() == "":
            end -= 1
        return {
            "file": relative_name,
            "name": class_node.name,
            "start": start,
            "end": end,
            "code": "\n".join(lines[start - 1 : end]),
        }

    found = []
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef)

    for node in tree.body:
        if isinstance(node, kinds):
            found.append(make_chunk(node, node.name))
        elif isinstance(node, ast.ClassDef):
            methods = [c for c in node.body if isinstance(c, kinds)]
            first_method = methods[0] if methods else None
            found.append(make_header_chunk(node, first_method))
            for child in methods:
                if not is_stub(child):
                    found.append(make_chunk(child, node.name + "." + child.name))
    return found


all_chunks = []
files_done = 0

for file_path in sorted(repo_path.rglob("*.py")):
    folders = file_path.relative_to(repo_path).parts
    if any(part in skip_folders for part in folders):
        continue
    try:
        all_chunks.extend(chunk_file(file_path))
        files_done += 1
    except (SyntaxError, UnicodeDecodeError):
        print("Skipped (could not read):", file_path)

print("Files processed:", files_done)
print("Total chunks:", len(all_chunks))

print()
print("Class header check:")
for chunk in all_chunks:
    if chunk["name"] == "Session":
        code_lines = chunk["code"].splitlines()
        print(chunk["start"], "to", chunk["end"])
        print(code_lines[0])
        print(code_lines[-1])

output_path = Path("data/chunks.json")
output_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
print()
print("Saved", len(all_chunks), "chunks to", output_path)