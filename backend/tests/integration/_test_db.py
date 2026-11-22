"""Shared, robust database-identity guard for integration tests.

Integration tests that need PostgreSQL must never run against a production
database by accident. This module centralises the safety checks so both the
uniqueness test and the ORM concurrency test use one policy.

Policy
------
* The target URL comes from ``HAORWATCH_TEST_DATABASE_URL``.
* A database name is *accepted* only when it is either:
  - explicitly listed in ``HAORWATCH_TEST_DB_ALLOWLIST`` (comma-separated
    database names), or
  - clearly a non-production test database by name (contains a safe marker such
    as ``test``/``ci``/``tmp``) **and** contains no protected marker.
* Protected markers include ``prod``, ``production``, ``live``, ``staging`` and
  the application database name ``haorwatch``.
* Unknown or ambiguous names (no safe marker and not allow-listed) are refused.
* A deliberate override ``HAORWATCH_ALLOW_NON_TEST_DB=1`` is required to run
  against any non-allow-listed target; it never makes a protected name pass.

Credentials are never logged or included in skip reasons; only the database name
is reported.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

# Substrings that unambiguously mark a database as production-like.
HARD_PROTECTED_MARKERS = ("prod", "production", "live", "staging")
# The application database name (docker-compose) is protected on its own, but its
# disposable test variant (e.g. ``haorwatch_test``) is allowed via a safe marker.
APP_DB_MARKERS = ("haorwatch",)
# All protected markers, for reporting.
PROTECTED_MARKERS = HARD_PROTECTED_MARKERS + APP_DB_MARKERS
# Substrings that clearly mark a disposable test database.
SAFE_MARKERS = ("test", "ci", "tmp", "temp", "scratch", "sandbox")


def database_name(url: str) -> str:
    """Extract the database name from a SQLAlchemy/psycopg URL (no credentials)."""
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


def host_port(url: str) -> tuple[str, int]:
    host = url.split("@")[-1].split("/")[0]
    hostname, _, port = host.partition(":")
    return (hostname or "localhost"), int(port or 5432)


def reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        hostname, port = host_port(url)
        sock = socket.create_connection((hostname, port), timeout=timeout)
        sock.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable here"
        return False


def allowlist() -> set[str]:
    raw = os.environ.get("HAORWATCH_TEST_DB_ALLOWLIST", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def allow_non_test_db() -> bool:
    return os.environ.get("HAORWATCH_ALLOW_NON_TEST_DB") == "1"


def classify(url: str) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a candidate URL's database name.

    ``reason`` never contains credentials. ``allowed`` is the *policy* decision
    independent of reachability.
    """
    name = database_name(url).strip()
    lowered = name.lower()

    if not name:
        return False, "could not determine a database name from HAORWATCH_TEST_DATABASE_URL"

    if lowered in allowlist():
        return True, f"database {lowered!r} is explicitly allow-listed"

    # Hard protected markers (prod/production/live/staging) never pass a safe
    # marker: ``prod_test`` is still a production target.
    hard = [marker for marker in HARD_PROTECTED_MARKERS if marker in lowered]
    if hard and not allow_non_test_db():
        return False, (
            f"database {lowered!r} matches protected marker(s) {hard}; "
            "refusing unless it is allow-listed via HAORWATCH_TEST_DB_ALLOWLIST "
            "or HAORWATCH_ALLOW_NON_TEST_DB=1 is set deliberately"
        )

    safe = [marker for marker in SAFE_MARKERS if marker in lowered]

    # The bare application database name (no safe marker) is protected.
    if (
        any(marker in lowered for marker in APP_DB_MARKERS)
        and not safe
        and not allow_non_test_db()
    ):
        return False, (
            f"database {lowered!r} is the application database name; refusing "
            "unless it is allow-listed via HAORWATCH_TEST_DB_ALLOWLIST "
            "or HAORWATCH_ALLOW_NON_TEST_DB=1 is set deliberately"
        )

    if not safe and not allow_non_test_db():
        return False, (
            f"database {lowered!r} is not recognisably a test database "
            f"(no marker in {SAFE_MARKERS}); add it to HAORWATCH_TEST_DB_ALLOWLIST "
            "to opt in explicitly"
        )

    return True, f"database {lowered!r} is a recognised test database"


@dataclass(frozen=True)
class DbTarget:
    url: str
    name: str
    available: bool
    allowed: bool
    reason: str

    @property
    def skip_reason(self) -> str | None:
        if not self.available:
            return "No reachable test database configured (set HAORWATCH_TEST_DATABASE_URL)."
        if not self.allowed:
            return f"Refused test database {self.name!r}: {self.reason}"
        return None


def resolve_target() -> DbTarget:
    url = os.environ.get("HAORWATCH_TEST_DATABASE_URL")
    if not url:
        return DbTarget(url="", name="", available=False, allowed=False, reason="no url configured")
    available = reachable(url)
    allowed, reason = classify(url)
    return DbTarget(url=url, name=database_name(url).lower(), available=available, allowed=allowed, reason=reason)


# --------------------------------------------------------------------------- #
# PostGIS / extension schema discovery
# --------------------------------------------------------------------------- #
def quote_ident(name: str) -> str:
    """Safely quote a locally-generated schema identifier."""
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe schema identifier: {name!r}")
    return '"' + name + '"'


def build_search_path(isolated_schema: str, extension_schemas: tuple[str, ...] = ()) -> str:
    """Build a quoted ``search_path`` value.

    Order: the unique isolated test schema first, then any extension schemas
    (e.g. the schema PostGIS is installed into), then ``public`` last so the
    isolated schema always wins for unqualified names. PostGIS is not assumed to
    live in ``public``. Identifiers are quoted and de-duplicated.
    """
    parts: list[str] = []
    for schema in (isolated_schema, *extension_schemas, "public"):
        if not schema:
            continue
        if schema not in parts:
            parts.append(schema)
    return ", ".join(quote_ident(part) for part in parts)


def discover_extension_schemas(engine, extensions: tuple[str, ...] = ("postgis",)) -> tuple[str, ...]:
    """Query the database for the schema(s) each extension is installed into.

    Returns the distinct extension schemas in a stable order. Falls back to
    ``("public",)`` when the catalog query yields nothing, since that is the
    PostgreSQL default for the PostGIS extension.
    """
    from sqlalchemy import text

    found: list[str] = []
    with engine.connect() as conn:
        for extension in extensions:
            try:
                schema = conn.execute(
                    text(
                        "SELECT n.nspname FROM pg_extension e "
                        "JOIN pg_namespace n ON n.oid = e.extnamespace "
                        "WHERE e.extname = :name"
                    ),
                    {"name": extension},
                ).scalar()
            except Exception:  # noqa: BLE001 - catalog unavailable; fall back below
                schema = None
            if schema and schema not in found:
                found.append(schema)
    if not found:
        return ("public",)
    return tuple(found)
