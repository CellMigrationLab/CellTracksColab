from collections import defaultdict
from pathlib import Path
import sys

NOTEBOOK_DIR = Path("notebooks").resolve()
README_PATH = NOTEBOOK_DIR / "README.md"


def discover_notebooks(directory: Path) -> list[Path]:
	return sorted(
		(path.relative_to(directory) for path in directory.rglob("*.ipynb")),
		key=lambda p: p.as_posix(),
	)


def build_readme_content(notebooks: list[Path]) -> str:
	lines = ["# Notebooks", ""]
	if not notebooks:
		lines.append("No notebooks found in this folder.")
		lines.append("")
		return "\n".join(lines)

	lines.append("Indexed list of folders and notebooks:")
	lines.append("")

	grouped: dict[Path, list[Path]] = defaultdict(list)
	for notebook in notebooks:
		grouped[notebook.parent].append(notebook)

	for folder, files in grouped.items():
		files.sort(key=lambda p: p.name.lower())

	for folder in sorted(grouped.keys(), key=lambda p: p.as_posix()):
		folder_label = "." if folder == Path(".") else folder.as_posix()
		lines.append(f"- [{folder_label}]({folder_label.replace(' ', '%20')}):")
		for notebook in grouped[folder]:
			lines.append(f"    - [{notebook.name}]({notebook.as_posix().replace(' ', '%20')})")
		lines.append("")

	return "\n".join(lines).rstrip() + "\n"


def main():
	if not NOTEBOOK_DIR.exists():
		print("notebooks directory not found.", file=sys.stderr)
		sys.exit(1)

	notebooks = discover_notebooks(NOTEBOOK_DIR)
	content = build_readme_content(notebooks)
	README_PATH.write_text(content, encoding="utf-8")
	print(f"Wrote {README_PATH} with {len(notebooks)} notebook(s).")


if __name__ == "__main__":
	main()
