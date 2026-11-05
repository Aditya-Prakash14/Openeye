#!/usr/bin/env python3
"""Compile every python code block in the course markdown.

Catches typos you would otherwise hit mid-exercise. Blocks labelled `bash`,
`text`, etc. are ignored — only ```python / ```py blocks are checked.

Usage:
    python scripts/check_solutions.py            # syntax-check all blocks
    python scripts/check_solutions.py --dump DIR # also write each block to DIR
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE_START = re.compile(r"^```(?:python|py)\s*$")


def python_blocks(text):
    """Yield (first_line_number, source) for each ```python fenced block."""
    inside, block, start = False, [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not inside and FENCE_START.match(line):
            inside, block, start = True, [], lineno + 1
        elif inside and line.strip() == "```":
            yield start, "\n".join(block)
            inside = False
        elif inside:
            block.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="directory to write each block to")
    args = ap.parse_args()

    checked = failures = 0
    for md in sorted(ROOT.rglob("*.md")):
        if ".git" in md.parts:
            continue
        for start, source in python_blocks(md.read_text()):
            checked += 1
            name = f"{md.relative_to(ROOT)}:{start}"
            if args.dump:
                dump = Path(args.dump)
                dump.mkdir(parents=True, exist_ok=True)
                safe = str(md.relative_to(ROOT)).replace("/", "__").replace(".md", "")
                (dump / f"{safe}__{start}.py").write_text(source + "\n")
            try:
                compile(source, name, "exec")
            except SyntaxError as e:
                failures += 1
                print(f"FAIL {name}: {e}")

    print(f"checked {checked} python blocks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
