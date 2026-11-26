import ast
import json
from pathlib import Path

repo_path = Path("data/repos/requests")
skip_folders = {".git", ".venv", "__pycache__", "tests", "docs"}


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
        if node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    def make_block_chunk(name, start, end):
        return {
            "file": relative_name,
            "name": name,
            "start": start,
            "end": end,
            "code": "\n".join(lines[start - 1 : end]),
        }

    def make_chunk(node, name):
        return make_block_chunk(name, first_line_of(node), node.end_lineno)

    def make_header_chunk(class_node, first_method):
        start = first_line_of(class_node)
        if first_method is None:
            end = class_node.end_lineno
        else:
            end = first_line_of(first_method) - 1
        while end > start and lines[end - 1].strip() == "":
            end -= 1
        return make_block_chunk(class_node.name, start, end)

    found = []
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
    assignments = (ast.Assign, ast.AnnAssign)

    # 1. The module docstring (the string at the very top of the file)
    if tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.append(make_block_chunk("module docstring", first.lineno, first.end_lineno))

    # 2. Runs of consecutive top-level assignments, like HOOKS = [...]
    run = []
    for node in tree.body + [None]:
        if isinstance(node, assignments):
            run.append(node)
            continue
        if run:
            names = []
            for item in run:
                target = item.targets[0] if isinstance(item, ast.Assign) else item.target
                names.append(ast.unparse(target))
            label = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
            found.append(make_block_chunk("constants: " + label, run[0].lineno, run[-1].end_lineno))
            run = []

    # 3. Functions, classes and methods, as before
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

    found.sort(key=lambda c: (c["start"], c["end"]))
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

old = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
old_keys = {(c["file"], c["name"], c["start"]) for c in old if not c["file"].startswith("docs/")}
new_keys = {(c["file"], c["name"], c["start"]) for c in all_chunks}

docstrings = [c for c in all_chunks if c["name"] == "module docstring"]
constants = [c for c in all_chunks if c["name"].startswith("constants: ")]

print("Files processed:", files_done)
print("Old chunks:", len(old), "  New chunks:", len(all_chunks))
print("Old chunks (outside docs/) missing from the new set:", len(old_keys - new_keys))
print("Docstring chunks:", len(docstrings), "  Constants chunks:", len(constants))

print()
print("5 biggest constants chunks:")
for c in sorted(constants, key=lambda c: c["end"] - c["start"], reverse=True)[:5]:
    print(f"  {c['end'] - c['start'] + 1:>3} lines  {c['file']}  {c['name']}")

print()
print("hooks.py chunks:")
for c in all_chunks:
    if c["file"].endswith("hooks.py"):
        print(f"  {c['start']}-{c['end']}  {c['name']}")

Path("data/chunks_v2.json").write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
print()
print("Saved", len(all_chunks), "chunks to data/chunks_v2.json")