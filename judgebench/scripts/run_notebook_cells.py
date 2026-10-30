import json
import sys

notebook_path = sys.argv[1]
start_cell = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end_cell = int(sys.argv[3]) if len(sys.argv) > 3 else 999

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

namespace = {
    "display": print
}

for cell_number, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    if not (start_cell <= cell_number <= end_cell):
        continue

    source = "".join(cell["source"])

    print(f"\n========== Running cell {cell_number} ==========\n")

    try:
        exec(
            compile(source, f"<cell {cell_number}>", "exec"),
            namespace
        )
    except Exception:
        print(f"\nCell {cell_number} failed.")
        raise

print("\nSelected notebook cells completed successfully.")