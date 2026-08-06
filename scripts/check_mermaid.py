"""Check Mermaid blocks in a markdown file for structural mistakes.

Catches the errors that are easy to make and hard to spot by eye — a typo in a
node name, a style applied to a box that doesn't exist. It does not check that
the diagram looks good; preview it for that.

Usage:
    python scripts/check_mermaid.py docs/pipeline.md
    python scripts/check_mermaid.py docs/*.md
"""
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)


def _strip_labels(body: str) -> str:
    """Blank out quoted label text.

    Labels routinely contain things that look like node syntax — a call such as
    ``converter.to_structured()`` would otherwise be read as a node definition.
    Replacing each label with a placeholder of the same shape keeps the
    surrounding structure intact.
    """
    return re.sub(r'"[^"]*"', '"_"', body)


def check_flowchart(body: str) -> list[str]:
    """Return a list of problems found in one flowchart block."""
    problems = []
    skeleton = _strip_labels(body)

    # A node exists if it is given a label — A["text"], A("text"), A{"text"} —
    # or if it simply appears on either end of an arrow. Both are valid Mermaid:
    # a bare name creates a node labelled with that name.
    labelled = set(re.findall(r"(\w+)\s*[\[({]", skeleton))
    on_arrow = set(re.findall(r"(\w+)\s*(?:-->|---|-\.->|==>)", skeleton))
    on_arrow |= set(
        re.findall(r"(?:-->|---|-\.->|==>)\s*(?:\|[^|]*\|\s*)?(\w+)", skeleton)
    )
    known = (labelled | on_arrow) - {"flowchart", "graph", "subgraph", "end"}
    # Style names are not nodes.
    known -= set(re.findall(r"classDef\s+(\w+)", skeleton))
    defined = known

    styles_defined = set(re.findall(r"classDef\s+(\w+)", skeleton))
    styles_used = set()
    for line in re.findall(r"^\s*class\s+([\w,]+)\s+(\w+)", skeleton, re.M):
        nodes, style = line
        styles_used.add(style)
        for n in nodes.split(","):
            if n and n not in defined:
                problems.append(f"class applied to undefined node {n!r}")

    for style in sorted(styles_used - styles_defined):
        problems.append(f"class uses style {style!r} that has no classDef")
    for style in sorted(styles_defined - styles_used):
        problems.append(f"classDef {style!r} is defined but never used (harmless)")

    # Unescaped angle brackets inside labels render as broken HTML.
    for label in re.findall(r'\["([^"]*)"\]', body):
        stripped = re.sub(r"</?(?:b|i|br|small|code|em|strong)\s*/?>", "", label)
        if "<" in stripped or ">" in stripped:
            problems.append(
                f"unescaped < or > in a label — use &lt; and &gt;: {label[:56]!r}"
            )

    return problems


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or [Path("docs/pipeline.md")]
    total_blocks = total_problems = 0

    for path in paths:
        if not path.exists():
            print(f"{path}: not found")
            return 1
        blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
        if not blocks:
            print(f"{path}: no mermaid blocks")
            continue

        for i, body in enumerate(blocks, 1):
            total_blocks += 1
            kind = body.strip().split()[0] if body.strip() else "?"
            label = f"{path} block {i} ({kind})"
            if kind not in {"flowchart", "graph"}:
                print(f"  {label}: not a flowchart — skipped")
                continue

            problems = check_flowchart(body)
            skeleton = _strip_labels(body)
            nodes = len(
                (set(re.findall(r"(\w+)\s*[\[({]", skeleton))
                 | set(re.findall(r"(\w+)\s*(?:-->|---|-\.->|==>)", skeleton))
                 | set(re.findall(r"(?:-->|---|-\.->|==>)\s*(?:\|[^|]*\|\s*)?(\w+)", skeleton)))
                - {"flowchart", "graph", "subgraph", "end"}
                - set(re.findall(r"classDef\s+(\w+)", skeleton)))
            if problems:
                total_problems += len(problems)
                print(f"  {label}: {len(problems)} problem(s), {nodes} nodes")
                for p in problems:
                    print(f"      - {p}")
            else:
                print(f"  {label}: OK — {nodes} nodes, nothing dangling")

    print()
    if total_problems:
        print(f"{total_problems} problem(s) across {total_blocks} block(s)")
        return 1
    print(f"all {total_blocks} block(s) look structurally sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
