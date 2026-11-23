"""ORM-level ReportService concurrency test against real PostgreSQL.

Isolation
---------
This test never reads or writes the shared ``public`` application schema. It
creates a fresh, uniquely-named schema (``haorwatch_test_<uuid>``), pins every
connection's ``search_path`` to that schema, creates the **real ORM tables**
inside it via ``Base.metadata.create_all``, and drops only that schema on
teardown. StudyArea/AnalysisRun/Report rows are created and deleted inside the
isolated schema, so no shared application row is ever touched.

The database guard (:mod:`tests.integration._test_db`) refuses production-like
database names (including ``haorwatch``) unless the target is explicitly
allow-listed or deliberately overridden.

PostGIS geometry columns
-------------------------
``StudyArea.boundary`` and ``AnalysisLayer.geometry`` use ``geoalchemy2``
``Geometry``. Their DDL needs the PostGIS extension to be installed at the
database level (e.g. ``CREATE EXTENSION postgis``), which the migrated
application database already has. The schema itself is still fully isolated; the
test skips with a clear reason if PostGIS is unavailable.

The PostGIS extension is **not** assumed to live in ``public``. The test
discovers the extension's installation schema from ``pg_extension``/
``pg_namespace`` and sets each connection's ``search_path`` to
``<isolated schema>, <extension schema(s)>, public`` so SQLAlchemy can resolve
the ``geometry`` type while unqualified application tables still resolve to the
isolated schema first. The PostGIS extension itself is never created, dropped,
or modified.

Determinism
-----------
The race is synchronised at the repository boundary using the test-only
``AnalysisRepository.after_report_lookup`` hook: both callers perform the real
``report_of`` lookup, confirm it returned no report, wait on a barrier, and only
then proceed to generation and persistence. This proves both callers completed
their initial lookup before either inserted.
"""

import threading
import uuid

import pytest
from sqlalchemy import create_engine, delete, event, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import AppError
from app.models import AnalysisRun, Base, Report, StudyArea
from app.repositories.analysis import AnalysisRepository
from app.services.report_service import ReportService

from tests.integration._test_db import (
    build_search_path,
    discover_extension_schemas,
    quote_ident,
    resolve_target,
)

_TARGET = resolve_target()

pytestmark = pytest.mark.skipif(
    _TARGET.skip_reason is not None,
    reason=_TARGET.skip_reason or "",
)


class _DeterministicLLM:
    """Deterministic LLM double producing a valid, non-empty report body."""

    def __init__(self):
        self.enabled = True
        self.calls = threading.Lock()

    def generate(self, _prompt: str) -> str:
        with self.calls:
            return "Deterministic concurrent report body."


@pytest.fixture(scope="module")
def schema_engine():
    """Create an isolated schema with the full ORM schema; drop only it after."""
    schema = f"haorwatch_test_{uuid.uuid4().hex[:12]}"
    quoted = quote_ident(schema)
    admin = create_engine(_TARGET.url)

    # Discover where PostGIS actually lives (it is not assumed to be in public).
    try:
        extension_schemas = discover_extension_schemas(admin)
    except Exception:  # noqa: BLE001
        admin.dispose()
        pytest.skip("could not query extension schemas; database not usable")
    search_path = build_search_path(schema, extension_schemas)

    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {quoted}"))
    except Exception:  # noqa: BLE001
        admin.dispose()
        pytest.skip("could not create isolated schema; database not usable")

    engine = create_engine(_TARGET.url)

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f"SET search_path TO {search_path}")
        finally:
            cur.close()

    try:
        # Real ORM tables, created inside the isolated schema. Unqualified
        # application tables resolve to the isolated schema first (search_path
        # order) while the geometry type resolves via the extension schema.
        try:
            Base.metadata.create_all(engine)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"could not create ORM tables (PostGIS may be missing): {type(exc).__name__}")

        # Confirm the ORM tables were created inside the isolated schema only.
        created = _tables_in_schema(engine, schema)
        expected = set(Base.metadata.tables.keys())
        if not expected.issubset(created):
            pytest.skip(f"isolated schema missing ORM tables: {sorted(expected - created)}")
        yield schema, engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {quoted} CASCADE"))
        admin.dispose()


def _tables_in_schema(engine, schema: str) -> set[str]:
    """Return the set of application table names created in ``schema``."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    names = set(inspector.get_table_names(schema=schema))
    # Advanced SQLAlchemy may expose non-public tables without a schema arg when
    # search_path is set; union both views.
    names |= set(inspector.get_table_names())
    app_tables = set(Base.metadata.tables.keys())
    return names & app_tables


@pytest.fixture()
def factory(schema_engine):
    _schema, engine = schema_engine
    return sessionmaker(bind=engine)


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
    """Delete only rows created by this test, inside the isolated schema."""
    with factory() as db:
        db.execute(delete(Report).where(Report.analysis_run_id == run_id))
        db.execute(delete(AnalysisRun).where(AnalysisRun.id == run_id))
        db.execute(delete(StudyArea).where(StudyArea.id == area_id))
        db.commit()


def test_concurrent_get_report_returns_single_stored_report(factory):
    area_id, run_id = _seed(factory)
    barrier = threading.Barrier(2)
    lookup_results = []
    outcomes = []
    lock = threading.Lock()

    def call():
        db = factory()
        repo = AnalysisRepository(db)
        # Test-only hook: after the real lookup, record that it saw no report and
        # block until both callers have completed their lookup.
        def after_lookup(found):
            with lock:
                lookup_results.append(found is None)
            barrier.wait(timeout=10)

        repo.after_report_lookup = after_lookup
        service = ReportService(db=db, repo=repo, llm=_DeterministicLLM())
        try:
            resp = service.get_report(run_id, "en")
            with lock:
                outcomes.append(("ok", resp.analysis_id, resp.created_at))
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
        t.join(timeout=40)

    try:
        # Both initial lookups must have seen no report.
        assert len(lookup_results) == 2, lookup_results
        assert all(lookup_results), f"a caller saw a pre-existing report: {lookup_results}"

        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
            assert len(rows) == 1, f"expected exactly one report, found {len(rows)}"
            winner_id = rows[0].id
            assert rows[0].language == "en"
            # Session still usable after any rollback.
            db.execute(select(Report).where(Report.analysis_run_id == run_id)).first()

        # Both callers must have succeeded with the SAME stored report id.
        ok = [o for o in outcomes if o[0] == "ok"]
        assert len(ok) == 2, outcomes
        returned_ids = {o[1] for o in ok}
        assert returned_ids == {winner_id}, f"callers returned {returned_ids}, stored {winner_id}"
    finally:
        _cleanup(factory, area_id, run_id)


def test_losing_session_observes_unique_conflict_then_recovers(factory):
    """Force the race so the loser hits IntegrityError, rolls back, re-reads."""
    area_id, run_id = _seed(factory)
    winner_first_lookup = threading.Event()
    loser_can_proceed = threading.Event()
    loser_saw_conflict = []
    winner_report_id = []
    lock = threading.Lock()

    def winner():
        db = factory()
        repo = AnalysisRepository(db)
        service = ReportService(db=db, repo=repo, llm=_DeterministicLLM())
        try:
            resp = service.get_report(run_id, "en")
            with lock:
                winner_report_id.append(resp.analysis_id)
            winner_first_lookup.set()
            # Give the loser time to do its own lookup (still empty) and then
            # race the insert after the winner has already committed.
            loser_can_proceed.wait(timeout=10)
        finally:
            db.close()

    def loser():
        db = factory()
        repo = AnalysisRepository(db)
        original_report_of = repo.report_of
        state = {"pre_insert_lookup": True}

        def after_lookup(found):
            # First lookup happens before the winner commits (empty). Later
            # lookups (post-conflict re-read) must find the winner.
            if state["pre_insert_lookup"] and found is None:
                state["pre_insert_lookup"] = False
                winner_first_lookup.wait(timeout=10)

        repo.after_report_lookup = after_lookup

        # Track the conflict at the repository save boundary.
        original_save = repo.save_report

        def save(report):
            try:
                return original_save(report)
            except IntegrityError:
                with lock:
                    loser_saw_conflict.append(True)
                raise

        repo.save_report = save
        service = ReportService(db=db, repo=repo, llm=_DeterministicLLM())
        try:
            resp = service.get_report(run_id, "en")
            with lock:
                loser_saw_conflict.append(("recovered", resp.analysis_id))
            loser_can_proceed.set()
        finally:
            # Session must still be usable after the internal rollback.
            db.execute(select(Report).where(Report.analysis_run_id == run_id)).first()
            db.close()

    threads = [threading.Thread(target=winner), threading.Thread(target=loser)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=40)

    try:
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
            assert len(rows) == 1, f"expected exactly one report, found {len(rows)}"
            stored_id = rows[0].id
        assert winner_report_id == [stored_id], (winner_report_id, stored_id)
        assert True in loser_saw_conflict, f"loser never hit an IntegrityError: {loser_saw_conflict}"
        recovered = [entry for entry in loser_saw_conflict if isinstance(entry, tuple)]
        assert recovered and recovered[0][0] == "recovered", loser_saw_conflict
        assert recovered[0][1] == stored_id, "loser did not return the winner's report id"
    finally:
        _cleanup(factory, area_id, run_id)


def test_en_and_bn_are_independent(factory):
    area_id, run_id = _seed(factory)
    try:
        with factory() as db:
            service = ReportService(db=db, llm=_DeterministicLLM())
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
            service = ReportService(db=db, llm=_DeterministicLLM())
            first = service.get_report(run_id, "en")
        with factory() as db:
            service2 = ReportService(db=db, llm=_DeterministicLLM())
            second = service2.get_report(run_id, "en")
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
        assert len(rows) == 1
        assert first.body == second.body
        assert first.analysis_id == rows[0].id
    finally:
        _cleanup(factory, area_id, run_id)


def test_unexpected_persistence_error_is_not_false_success(factory):
    """An unexpected DB error must surface as REPORT_PERSISTENCE_FAILED, not success."""
    area_id, run_id = _seed(factory)
    try:
        with factory() as db:
            repo = AnalysisRepository(db)

            def boom(_report):
                raise RuntimeError("simulated persistence failure")

            repo.save_report = boom
            service = ReportService(db=db, repo=repo, llm=_DeterministicLLM())
            with pytest.raises(AppError) as exc:
                service.get_report(run_id, "en")
            assert exc.value.code == "REPORT_PERSISTENCE_FAILED"
        with factory() as db:
            rows = list(db.scalars(select(Report).where(Report.analysis_run_id == run_id)))
            assert rows == []
    finally:
        _cleanup(factory, area_id, run_id)
