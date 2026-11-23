"""Real-PostgreSQL verification of the reports uniqueness constraint.

Runs only when a reachable PostgreSQL database is configured via
HAORWATCH_TEST_DATABASE_URL. Skipped otherwise so the unit suite needs no DB.

SAFETY: this test never touches shared/public application tables. It creates a
fresh, uniquely-named schema (``haorwatch_test_<uuid>``), creates its minimal
test tables inside that schema, and drops only that schema on teardown. All
connections set ``search_path`` to the isolated schema. It refuses to run
against a database that looks like production unless explicitly overridden.
"""

import threading
import uuid

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError

from tests.integration._test_db import resolve_target

_TARGET = resolve_target()
TEST_URL = _TARGET.url

pytestmark = pytest.mark.skipif(
    _TARGET.skip_reason is not None,
    reason=_TARGET.skip_reason or "",
)


def _ident(name: str) -> str:
    """Safely quote a locally-generated schema identifier (uuid-derived)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe schema identifier: {name!r}")
    return '"' + name + '"'


@pytest.fixture(scope="module")
def isolated_schema():
    """Create and drop a uniquely-named schema; never touch public tables."""
    schema = f"haorwatch_test_{uuid.uuid4().hex[:12]}"
    quoted = _ident(schema)
    admin = create_engine(TEST_URL)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {quoted}"))
    try:
        yield schema
    finally:
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {quoted} CASCADE"))
        admin.dispose()


def _schema_engine(schema: str):
    """Engine whose every connection pins search_path to the isolated schema."""
    quoted = _ident(schema)
    eng = create_engine(TEST_URL)

    @event.listens_for(eng, "connect")
    def _set_search_path(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f"SET search_path TO {quoted}")
        finally:
            cur.close()

    return eng


TEST_TABLES = [
    "CREATE TABLE report_test_areas (id uuid PRIMARY KEY, slug text, "
    "name text, country_code text DEFAULT 'BD')",
    "CREATE TABLE analysis_runs (id uuid PRIMARY KEY, "
    "study_area_id uuid REFERENCES report_test_areas(id) ON DELETE CASCADE, "
    "status text NOT NULL DEFAULT 'queued', analysis_label text NOT NULL, "
    "processing_config jsonb NOT NULL DEFAULT '{}')",
    "CREATE TABLE reports (id uuid PRIMARY KEY, "
    "analysis_run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE, "
    "language varchar(8) NOT NULL, generation_method varchar(24) NOT NULL, "
    "status varchar(24) NOT NULL, title varchar(180) NOT NULL, body text NOT NULL, "
    "source_snapshot jsonb, created_at timestamptz NOT NULL DEFAULT now(), "
    "CONSTRAINT uq_report_run_language UNIQUE (analysis_run_id, language))",
]


@pytest.fixture(scope="module")
def engine(isolated_schema):
    eng = _schema_engine(isolated_schema)
    with eng.begin() as conn:
        for stmt in TEST_TABLES:
            conn.execute(text(stmt))
    yield eng
    eng.dispose()


@pytest.fixture()
def seeded(engine):
    area_id = uuid.uuid4()
    run_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO report_test_areas (id, slug, name) VALUES (:i,'a','A')"),
            {"i": area_id},
        )
        conn.execute(
            text("INSERT INTO analysis_runs (id, study_area_id, analysis_label) VALUES (:r,:a,'run')"),
            {"r": run_id, "a": area_id},
        )
    return run_id


def _insert_report(conn, run_id, language):
    return conn.execute(
        text(
            "INSERT INTO reports (id, analysis_run_id, language, generation_method, status, title, body) "
            "VALUES (gen_random_uuid(), :r, :l, 'template', 'completed', 't', 'b')"
        ),
        {"r": run_id, "l": language},
    )


def test_same_run_same_language_rejected(engine, seeded):
    with engine.begin() as conn:
        _insert_report(conn, seeded, "en")
    with pytest.raises(IntegrityError) as exc:
        with engine.begin() as conn:
            _insert_report(conn, seeded, "en")
    assert "uq_report_run_language" in str(exc.value.orig)


def test_same_run_different_language_allowed(engine, seeded):
    with engine.begin() as conn:
        _insert_report(conn, seeded, "en")
        _insert_report(conn, seeded, "bn")
        count = conn.execute(
            text("SELECT count(*) FROM reports WHERE analysis_run_id = :r"), {"r": seeded}
        ).scalar()
    assert count == 2


def test_different_run_same_language_allowed(engine, seeded):
    other_run = uuid.uuid4()
    with engine.begin() as conn:
        area = conn.execute(
            text("SELECT study_area_id FROM analysis_runs WHERE id = :r"), {"r": seeded}
        ).scalar()
        conn.execute(
            text("INSERT INTO analysis_runs (id, study_area_id, analysis_label) VALUES (:r,:a,'run2')"),
            {"r": other_run, "a": area},
        )
        _insert_report(conn, seeded, "en")
        _insert_report(conn, other_run, "en")
        count = conn.execute(text("SELECT count(*) FROM reports WHERE language='en'")).scalar()
    assert count >= 2


def test_concurrent_inserts_yield_exactly_one_row(isolated_schema, engine, seeded):
    """Two real connections race; exactly one row must exist afterwards."""
    barrier = threading.Barrier(2)
    results = []

    def worker():
        # Separate engine/connection per thread, same isolated schema.
        local = _schema_engine(isolated_schema)
        try:
            barrier.wait(timeout=10)
            with local.begin() as conn:
                _insert_report(conn, seeded, "en")
                results.append("ok")
        except IntegrityError:
            results.append("conflict")
        finally:
            local.dispose()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    with engine.begin() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM reports WHERE analysis_run_id = :r AND language = 'en'"),
            {"r": seeded},
        ).scalar()
        # Session must still be usable after any rollback (no aborted transaction).
        conn.execute(text("SELECT 1"))

    assert count == 1
    assert results.count("ok") >= 1
    assert len(results) == 2
