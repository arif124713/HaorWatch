"""ORM-level ReportService concurrency test against real PostgreSQL.

Requires a migrated schema (run ``alembic upgrade head`` first) and a reachable
database via HAORWATCH_TEST_DATABASE_URL. Skipped otherwise.

Uses the real ORM models and the real ReportService.get_report() conflict path
(IntegrityError -> rollback -> re-read winner) with two independent sessions
racing on the same (analysis_run_id, language). Rows are cleaned up afterwards;
no shared/public data is dropped.
"""

import os
import socket
import threading
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import AppError
from app.models import AnalysisRun, Report, StudyArea
from app.repositories.analysis import AnalysisRepository
from app.services.report_service import ReportService

TEST_URL = os.environ.get("HAORWATCH_TEST_DATABASE_URL")
ALLOW_NON_TEST_DB = os.environ.get("HAORWATCH_ALLOW_NON_TEST_DB") == "1"
_PRODUCTION_MARKERS = ("prod", "production", "live", "staging")
_SAFE_MARKERS = ("test", "ci", "tmp", "scratch", "sandbox")


def _database_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?", 1)[0].lower()


def _reachable(url: str) -> bool:
    try:
        host = url.split("@")[-1].split("/")[0]
        hostname, _, port = host.partition(":")
        s = socket.create_connection((hostname or "localhost", int(port or 5432)), timeout=2)
        s.close()
        return True
    except Exception:
        return False


def _looks_like_production(url: str) -> bool:
    name = _database_name(url)
    if any(m in name for m in _SAFE_MARKERS):
        return False
    return any(m in name for m in _PRODUCTION_MARKERS)


_AVAILABLE = bool(TEST_URL) and _reachable(TEST_URL) if TEST_URL else False
_REFUSED = _AVAILABLE and _looks_like_production(TEST_URL) and not ALLOW_NON_TEST_DB

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="No reachable test database configured (set HAORWATCH_TEST_DATABASE_URL).",
)


@pytest.fixture()
def factory():
    if _REFUSED:
        pytest.skip("Database looks like production; set HAORWATCH_ALLOW_NON_TEST_DB=1 to override.")
    engine = create_engine(TEST_URL)
    try:
        with engine.connect() as conn:
            ok = conn.exec_driver_sql("SELECT to_regclass('public.reports') IS NOT NULL").scalar()
        if not ok:
            pytest.skip("reports table not present; run `alembic upgrade head` first")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not usable: {type(exc).__name__}")
    yield sessionmaker(bind=engine)
    engine.dispose()


def _seed(factory):
    with factory() as db:
        area = StudyArea(slug=f"test-{uuid.uuid4().hex[:8]}", name="Test Haor", country_code="BD")
        db.add(area)
        db.flush()
        run = AnalysisRun(study_area_id=area.id, status="completed", analysis_label="concurrency test")
        db.add(run)
        db.commit()
        return area.id, run.id


def _cleanup(factory, area_id, run_id):
    with factory() as db:
        db.execute(select(Report).where(Report.analysis_run_id == run_id))  # noqa: F841
        from sqlalchemy import delete

        db.execute(delete(Report).where(Report.analysis_run_id == run_id))
        db.execute(delete(AnalysisRun).where(AnalysisRun.id == run_id))
        db.execute(delete(StudyArea).where(StudyArea.id == area_id))
        db.commit()


def test_concurrent_get_report_returns_single_stored_report(factory):
    area_id, run_id = _seed(factory)
    barrier = threading.Barrier(2)
    outcomes = []
    lock = threading.Lock()

    def call():
        db = factory()
        try:
            service = ReportService(db=db)
            # Ensure both threads pass the "no existing report" lookup before either saves.
            barrier.wait(timeout=10)
            resp = service.get_report(run_id, "en")
            with lock:
                outcomes.append(("ok", str(resp.body)[:0], str(resp.created_at)))
        except AppError as exc:
            with lock:
                outcomes.append(("apperror", exc.code, None))
        except Exception as exc:  # noqa: BLE001
            with lock:
                outcomes.append(("error", type(exc).__name__, None))
        finally:
            db.close()

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    try:
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
            assert len(rows) == 1, f"expected exactly one report, found {len(rows)}"
            assert rows[0].language == "en"
            # Session still usable.
            db.execute(select(Report).where(Report.analysis_run_id == run_id)).first()
        # Both callers must have succeeded (winner + loser re-read winner).
        assert [o[0] for o in outcomes].count("ok") == 2, outcomes
    finally:
        _cleanup(factory, area_id, run_id)


def test_en_and_bn_are_independent(factory):
    area_id, run_id = _seed(factory)
    try:
        with factory() as db:
            service = ReportService(db=db)
            service.get_report(run_id, "en")
            service.get_report(run_id, "bn")
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
            langs = sorted(r.language for r in rows)
            assert langs == ["bn", "en"]
    finally:
        _cleanup(factory, area_id, run_id)


def test_second_request_reuses_stored_report(factory):
    area_id, run_id = _seed(factory)
    try:
        with factory() as db:
            service = ReportService(db=db)
            first = service.get_report(run_id, "en")
        with factory() as db:
            service2 = ReportService(db=db)
            second = service2.get_report(run_id, "en")
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
        assert len(rows) == 1
        assert first.body == second.body
    finally:
        _cleanup(factory, area_id, run_id)
