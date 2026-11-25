import ast
from pathlib import Path

file_path = Path("data/repos/requests/src/requests/models.py")
source = file_path.read_text(encoding="utf-8")
tree = ast.parse(source)

count = 0
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.decorator_list:
                decorator_line = child.decorator_list[0].lineno
                print(node.name + "." + child.name)
                print("  decorator is on line:", decorator_line)
                print("  chunker says it starts on line:", child.lineno)
                count += 1
    if count >= 5:
        break