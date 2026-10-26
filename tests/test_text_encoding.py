"""Every text read/write must name its encoding.

Windows defaults to cp1252, so a report containing "≥" crashed the sync-vs-async
benchmark after a 10-minute run on the RTX 3080 laptop. Linux CI never sees it
because its default is UTF-8, so check the source instead.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _offenders():
    for d in ("kiln", "scripts", "examples", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name not in ("read_text", "write_text", "open"):
                    continue
                if (
                    isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name)
                    and f.value.id in ("os", "tarfile", "Image", "gzip")
                ):
                    continue
                mode = next(
                    (a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)),
                    "",
                )
                if "b" in mode or any(k.arg == "encoding" for k in node.keywords):
                    continue
                if (
                    name == "open"
                    and isinstance(f, ast.Name)
                    and node.args
                    and not mode
                    and len(node.args) > 1
                ):
                    continue
                yield f"{p.relative_to(ROOT)}:{node.lineno} {name}"


def test_all_text_io_specifies_encoding():
    bad = list(_offenders())
    assert not bad, "missing encoding=:\n" + "\n".join(bad)
