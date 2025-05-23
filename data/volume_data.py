import os

def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if not f.startswith("."):
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)

root_dir = "."  # <- asegúrate de que sea "." si estás ya en data/
summary = []

print(f"📂 Explorando contenido dentro de: {os.path.abspath(root_dir)}\n")

for dirpath, dirnames, filenames in os.walk(root_dir):
    visible_files = [f for f in filenames if not f.startswith('.')]
    if visible_files:
        rel_path = os.path.relpath(dirpath, root_dir)
        rel_path = f"data/{rel_path}" if rel_path != "." else "data/"
        summary.append((rel_path, len(visible_files), get_size(dirpath)))

# Imprimir tabla final
print(f"\n{'Directory':<50} {'# Files':<10} {'Size (MB)':<10}")
print("-" * 75)
for path, num_files, size in sorted(summary):
    print(f"{path:<50} {num_files:<10} {size:<10}")
