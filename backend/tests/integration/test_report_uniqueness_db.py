"""Database-backed report uniqueness test.

Runs only when a reachable database is configured (HAORWATCH_TEST_DATABASE_URL
or the default DATABASE_URL). Skipped otherwise, so the unit suite does not
require Postgres.
"""

import os
import socket
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker


def _db_url() -> str | None:
    url = os.environ.get("HAORWATCH_TEST_DATABASE_URL")
    if url:
        return url
    try:
        from app.core.config import get_settings

        return get_settings().database_url
    except Exception:
        return None


def _reachable(url: str) -> bool:
    try:
        host = url.split("@")[-1].split("/")[0]
        hostname, _, port = host.partition(":")
        s = socket.create_connection((hostname or "localhost", int(port or 5432)), timeout=2)
        s.close()
        return True
    except Exception:
        return False


URL = _db_url()
pytestmark = pytest.mark.skipif(
    not URL or not _reachable(URL),
    reason="No reachable test database configured (set HAORWATCH_TEST_DATABASE_URL).",
)


@pytest.fixture()
def session():
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.rollback()
        db.close()
        engine.dispose()


def _ensure_schema(db):
    # Minimal slice needed for the constraint test; skip if reports table absent.
    exists = db.execute(
        text("SELECT to_regclass('public.reports') IS NOT NULL")
    ).scalar()
    if not exists:
        pytest.skip("reports table not present; run migrations first")


def test_unique_constraint_rejects_duplicate_report(session):
    _ensure_schema(session)
    study_area_id = uuid.uuid4()
    run_id = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO study_areas (id, slug, name, country_code) "
            "VALUES (:id, :slug, :name, 'BD')"
        ),
        {"id": study_area_id, "slug": f"test-{study_area_id}", "name": "Test"},
    )
    session.execute(
        text(
            "INSERT INTO analysis_runs (id, study_area_id, status, analysis_label, processing_config) "
            "VALUES (:id, :area, 'completed', 'Test run', '{}')"
        ),
        {"id": run_id, "area": study_area_id},
    )
    insert = text(
        "INSERT INTO reports (id, analysis_run_id, language, generation_method, status, title, body) "
        "VALUES (:id, :run, :lang, 'template', 'completed', 't', 'b')"
    )
    session.execute(insert, {"id": uuid.uuid4(), "run": run_id, "lang": "en"})
    session.commit()

    with pytest.raises(IntegrityError):
        session.execute(insert, {"id": uuid.uuid4(), "run": run_id, "lang": "en"})
        session.commit()
    session.rollback()

    # The original row must remain intact.
    count = session.execute(
        text("SELECT count(*) FROM reports WHERE analysis_run_id = :run AND language = 'en'"),
        {"run": run_id},
    ).scalar()
    assert count == 1

    # Different language is allowed.
    session.execute(insert, {"id": uuid.uuid4(), "run": run_id, "lang": "bn"})
    session.commit()
    count_bn = session.execute(
        text("SELECT count(*) FROM reports WHERE analysis_run_id = :run AND language = 'bn'"),
        {"run": run_id},
    ).scalar()
    assert count_bn == 1

    # Clean up.
    session.execute(text("DELETE FROM reports WHERE analysis_run_id = :run"), {"run": run_id})
    session.execute(text("DELETE FROM analysis_runs WHERE id = :id"), {"id": run_id})
    session.execute(text("DELETE FROM study_areas WHERE id = :id"), {"id": study_area_id})
    session.commit()
