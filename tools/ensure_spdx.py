# SPDX-License-Identifier: MPL-2.0
"""Ensure SPDX-License-Identifier: MPL-2.0 headers exist in tracked files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

LICENSE_LINE = "SPDX-License-Identifier: MPL-2.0"

COMMENT_STYLES = {
    ".py": "#",
    ".pyi": "#",
    ".pyx": "#",
    ".pxd": "#",
    ".pxi": "#",
    ".yml": "#",
    ".yaml": "#",
    ".toml": "#",
    ".cfg": "#",
    ".ini": "#",
    ".conf": "#",
    ".txt": "#",
    ".md": "<!--",
    ".rst": "..",
    ".json": "//",
    ".gitignore": "#",
}

SPECIAL_FILENAMES = {
    "dockerfile": "#",
}

SKIP_PATHS = {
    "LICENSE",
}


def iter_files(paths: Iterable[str]) -> Iterable[Path]:
    for name in paths:
        path = Path(name)
        if not path.exists():
            continue
        if path.name in SKIP_PATHS:
            continue
        if path.suffix == ".ipynb":
            yield path
            continue
        if path.is_dir():
            for sub in path.rglob("*"):
                if sub.is_file():
                    yield sub
            continue
        yield path


def ensure_notebook_header(path: Path) -> bool:
    data = json.loads(path.read_text())
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"<!-- {LICENSE_LINE} -->"],
    }
    cells = data.get("cells", [])
    if cells:
        first = cells[0]
        if (
            first.get("cell_type") == "markdown"
            and any(LICENSE_LINE in part for part in first.get("source", []))
        ):
            return False
        data["cells"] = [header_cell] + cells
    else:
        data["cells"] = [header_cell]
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return True


def ensure_header(path: Path) -> bool:
    suffix = path.suffix.lower()
    comment = COMMENT_STYLES.get(suffix)
    if comment is None:
        comment = SPECIAL_FILENAMES.get(path.name.lower())
    if comment is None:
        return False
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return False
    if LICENSE_LINE in "\n".join(text.splitlines()[:5]):
        return False
    if comment == "<!--":
        path.write_text(f"<!-- {LICENSE_LINE} -->\n" + text)
        return True
    if comment == "..":
        path.write_text(f".. {LICENSE_LINE}\n" + text)
        return True
    if comment == "//":
        path.write_text(f"// {LICENSE_LINE}\n" + text)
        return True
    if comment == "#":
        lines = text.splitlines()
        if lines and lines[0].startswith("#!"):
            shebang = lines[0]
            rest = "\n".join(lines[1:])
            new_text = shebang + "\n" + f"# {LICENSE_LINE}\n"
            if rest:
                new_text += rest
            path.write_text(new_text)
        else:
            path.write_text(f"# {LICENSE_LINE}\n" + text)
        return True
    return False


def main(argv: list[str]) -> int:
    modified = False
    for path in iter_files(argv or ["."]):
        if path.suffix == ".ipynb":
            modified |= ensure_notebook_header(path)
        else:
            modified |= ensure_header(path)
    if modified:
        print("Inserted missing SPDX license headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
