"""Structural safety guard for the test suite.

Fails if any test module contains destructive SQL against unqualified/shared
application tables. Integration tests must isolate their own schema instead of
dropping public tables.
"""

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

# Executable destructive patterns we never want in tests. These are matched
# against lowercased source text.
FORBIDDEN = [
    r"drop\s+table\s+(if\s+exists\s+)?(reports|analysis_runs|analysis_summaries|"
    r"analysis_metrics|analysis_layers|study_areas|dataset_assets)\b",
    r"drop\s+schema\s+(if\s+exists\s+)?public\b",
    r"truncate\s+(table\s+)?(reports|analysis_runs|study_areas)\b",
]

# The uniqueness test references these strings only inside assertions/comments.
_ALLOWLIST_LINE_MARKERS = (
    "forbidden",
    '"drop table',
    "'drop table",
    '"drop schema public"',
    "'drop schema public'",
    "structural guard",
)


def _test_files():
    return [p for p in TESTS_ROOT.rglob("test_*.py") if p.name != Path(__file__).name]


def test_no_destructive_public_schema_sql_in_tests():
    offenders = []
    for path in _test_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if any(marker in lowered for marker in _ALLOWLIST_LINE_MARKERS):
                continue
            for pattern in FORBIDDEN:
                if re.search(pattern, lowered):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == [], "Destructive public-schema SQL found in tests:\n" + "\n".join(offenders)
