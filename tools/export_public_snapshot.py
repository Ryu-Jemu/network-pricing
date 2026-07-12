"""Create a code-only public snapshot of this repository.

Per release policy, the public GitHub repository contains CODE ONLY:
no trained models, no result JSONs, no training logs, no dashboards.
This script copies the publishable subset into a target directory and
writes a public .gitignore so accidental re-addition is blocked.

Cross-platform (pathlib/shutil only).

Usage:
    python tools/export_public_snapshot.py ../network-pricing-public
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "src",
    "tests",
    "tools",
    ".github",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache"}

PUBLIC_GITIGNORE = """\
__pycache__/
*.pyc
.pytest_cache/
# experiment artefacts are not published (code-only policy)
models/
results/
training_logs/
*.zip
*.html
"""


def copy_tree(src: Path, dst: Path):
    for item in src.rglob("*"):
        if any(part in EXCLUDE_DIR_NAMES for part in item.parts):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="destination directory")
    args = ap.parse_args()
    dst = Path(args.target).resolve()
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"refusing to export into non-empty {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    for name in INCLUDE:
        src = ROOT / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        if src.is_dir():
            copy_tree(src, dst / name)
        else:
            shutil.copy2(src, dst / name)
        print(f"  copied: {name}")

    (dst / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    print(f"\npublic snapshot at {dst}")
    print("next: cd there, `git init && git add -A && git commit`, "
          "review `git status` before pushing.")


if __name__ == "__main__":
    main()
