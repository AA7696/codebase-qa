import sys
from pathlib import Path

print("Python:", sys.version.split()[0])
print("Inside venv:", sys.prefix != sys.base_prefix)

repo = Path("data/repos/requests")
py_files = list(repo.rglob("*.py"))
print("requests repo found:", repo.exists())
print("Python files in requests:", len(py_files))