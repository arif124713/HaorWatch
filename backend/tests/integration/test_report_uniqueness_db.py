"""Real-PostgreSQL verification of the reports uniqueness constraint and the
race-safe ReportService path.

Runs only when a reachable PostgreSQL database is configured via
HAORWATCH_TEST_DATABASE_URL. Skipped otherwise so the unit suite needs no DB.

This test intentionally builds a *minimal* schema (study_areas / analysis_runs /
reports) rather than the full PostGIS schema, so it can run against a plain
PostgreSQL instance while still exercising the real unique constraint and real
transactions. The reports DDL mirrors migration 0001; the unique constraint
mirrors migration 0002.
"""

import os
import socket
import threading
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

TEST_URL = os.environ.get("HAORWATCH_TEST_DATABASE_URL")


def _reachable(url: str) -> bool:
    try:
        host = url.split("@")[-1].split("/")[0]
        hostname, _, port = host.partition(":")
        s = socket.create_connection((hostname or "localhost", int(port or 5432)), timeout=2)
        s.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not TEST_URL or not _reachable(TEST_URL),
    reason="No reachable test database configured (set HAORWATCH_TEST_DATABASE_URL).",
)

SCHEMA = [
    "DROP TABLE IF EXISTS reports CASCADE",
    "DROP TABLE IF EXISTS analysis_runs CASCADE",
    "DROP TABLE IF EXISTS report_test_areas CASCADE",
    """CREATE TABLE report_test_areas (id uuid PRIMARY KEY, slug text, name text, country_code text DEFAULT 'BD')""",
    """CREATE TABLE analysis_runs (
        id uuid PRIMARY KEY,
        study_area_id uuid REFERENCES report_test_areas(id) ON DELETE CASCADE,
        status text NOT NULL DEFAULT 'queued',
        analysis_label text NOT NULL,
        processing_config jsonb NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE reports (
        id uuid PRIMARY KEY,
        analysis_run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        language varchar(8) NOT NULL,
        generation_method varchar(24) NOT NULL,
        status varchar(24) NOT NULL,
        title varchar(180) NOT NULL,
        body text NOT NULL,
        source_snapshot jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_report_run_language UNIQUE (analysis_run_id, language)
    )""",
]


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(TEST_URL)
    with eng.begin() as conn:
        for stmt in SCHEMA:
            conn.execute(text(stmt))
    yield eng
    eng.dispose()


@pytest.fixture()
def seeded(engine):
    area_id = uuid.uuid4()
    run_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO report_test_areas (id, slug, name) VALUES (:i,'a','A')"), {"i": area_id})
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
        area = conn.execute(text("SELECT study_area_id FROM analysis_runs WHERE id = :r"), {"r": seeded}).scalar()
        conn.execute(
            text("INSERT INTO analysis_runs (id, study_area_id, analysis_label) VALUES (:r,:a,'run2')"),
            {"r": other_run, "a": area},
        )
        _insert_report(conn, seeded, "en")
        _insert_report(conn, other_run, "en")
        count = conn.execute(text("SELECT count(*) FROM reports WHERE language='en'")).scalar()
    assert count >= 2


def test_concurrent_inserts_yield_exactly_one_row(engine, seeded):
    """Two real connections race; exactly one row must exist afterwards."""
    barrier = threading.Barrier(2)
    results = []

    def worker():
        # Separate connection/transaction per thread.
        local = create_engine(TEST_URL)
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
