#!/usr/bin/env python3
"""Extract each exercise's solution into a standalone runnable file.

The exercise markdown is the single source of truth; `solutions/*.py` are generated
from it. Re-run after editing any exercise:

    python scripts/extract_solutions.py          # write solutions/
    python scripts/extract_solutions.py --check  # fail if solutions/ is stale
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"
SOLUTIONS = ROOT / "solutions"
FENCE = re.compile(r"```python\n(.*?)```", re.S)


def title_of(md_text):
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify solutions are up to date")
    args = ap.parse_args()

    stale, written = [], []
    for md in sorted(EXERCISES.glob("*.md")):
        text = md.read_text()
        match = FENCE.search(text)
        if not match:
            raise SystemExit(f"no python block found in {md}")
        source = (f"# {title_of(text)} — solution\n"
                  f"# Generated from {md.relative_to(ROOT)} by scripts/extract_solutions.py.\n"
                  f"# Run:  python solutions/{md.stem}.py   (writes output files to the cwd)\n\n"
                  f"{match.group(1).strip()}\n")
        target = SOLUTIONS / f"{md.stem}.py"
        if target.exists() and target.read_text() == source:
            continue
        stale.append(target.name)
        if not args.check:
            target.parent.mkdir(exist_ok=True)
            target.write_text(source)
            written.append(target.name)

    if args.check:
        if stale:
            print(f"stale solutions (re-run scripts/extract_solutions.py): {', '.join(stale)}")
            return 1
        print(f"solutions are up to date ({len(list(SOLUTIONS.glob('*.py')))} files)")
        return 0

    print(f"wrote {len(written)} file(s) to solutions/: {', '.join(written) if written else 'nothing to do'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
