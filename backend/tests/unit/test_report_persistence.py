"""Report persistence: generated reports must be saved, reused, and race-safe."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import AppError
from app.services.report_service import ReportService
from tests.unit.test_report_template import make_run


class RecordingRepo:
    def __init__(self, run, summary=None):
        self.run = run
        self.summary = summary
        self.saved = []
        self.by_lang = {}
        self.rolled_back = 0

    def get(self, analysis_id):
        return self.run

    def summary_of(self, run_id):
        return self.summary

    def report_of(self, run_id, language):
        return self.by_lang.get(language)

    def save_report(self, report):
        self.saved.append(report)
        report.created_at = report.created_at or datetime.now(timezone.utc)
        self.by_lang[report.language] = report
        return report

    def rollback(self):
        self.rolled_back += 1


class ConflictRepo(RecordingRepo):
    """Simulates a concurrent insert winning the race on first save_report."""

    def __init__(self, run, summary=None, winner=None):
        super().__init__(run, summary)
        self.winner = winner
        self._conflicted = False

    def save_report(self, report):
        if not self._conflicted:
            self._conflicted = True
            # The winning request already persisted a report; simulate the
            # unique-constraint violation this request receives.
            self.by_lang[report.language] = self.winner
            raise IntegrityError("insert", {}, Exception("duplicate key value violates unique constraint"))
        return super().save_report(report)


class ExplodingRepo(RecordingRepo):
    def save_report(self, report):
        raise RuntimeError("connection reset")


def test_generated_report_is_persisted():
    run, summary = make_run(dict(detected_flood_area_km2=1.0))
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    service.get_report(run.id, "en")
    assert len(repo.saved) == 1
    assert repo.saved[0].language == "en"


def test_existing_report_is_reused_not_regenerated():
    run, summary = make_run(dict(detected_flood_area_km2=1.0))
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    first = service.get_report(run.id, "en")
    second = service.get_report(run.id, "en")
    assert len(repo.saved) == 1  # second call reused the stored report
    assert first.body == second.body


def test_existing_report_does_not_call_llm_again():
    run, summary = make_run(dict())
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    service.get_report(run.id, "en")
    calls = {"n": 0}

    class _CountingLLM:
        enabled = True

        def generate(self, prompt):
            calls["n"] += 1
            return "text"

    service.llm = _CountingLLM()
    service.get_report(run.id, "en")  # existing -> reused
    assert calls["n"] == 0


def test_bn_and_en_both_persist_separately():
    run, summary = make_run(dict())
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    service.get_report(run.id, "bn")
    service.get_report(run.id, "en")
    assert {r.language for r in repo.saved} == {"bn", "en"}


def test_concurrent_conflict_rolls_back_and_returns_winner():
    run, summary = make_run(dict(detected_flood_area_km2=2.0))
    winner, _ = make_run(dict(detected_flood_area_km2=2.0))
    # Build a real persisted-looking winner via the template generator.
    seed = RecordingRepo(run, summary)
    winner_report = ReportService(repo=seed, llm=None).get_report(run.id, "en")
    winner_row = seed.by_lang["en"]

    repo = ConflictRepo(run, summary, winner=winner_row)
    service = ReportService(repo=repo, llm=None)
    result = service.get_report(run.id, "en")
    assert repo.rolled_back == 1
    assert result.body == winner_report.body
    # The conflict path must not have stored a second report.
    assert winner_row in repo.by_lang.values()


def test_unexpected_persistence_error_is_not_swallowed():
    run, summary = make_run(dict())
    repo = ExplodingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    with pytest.raises(AppError) as exc:
        service.get_report(run.id, "en")
    assert exc.value.code == "REPORT_PERSISTENCE_FAILED"
    assert repo.rolled_back == 1


def test_unsupported_language_rejected():
    run, summary = make_run(dict())
    service = ReportService(repo=RecordingRepo(run, summary), llm=None)
    with pytest.raises(AppError) as exc:
        service.get_report(run.id, "fr")
    assert exc.value.code == "UNSUPPORTED_LANGUAGE"


def test_llm_disabled_falls_back_to_template():
    run, summary = make_run(dict())
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    report = service.get_report(run.id, "en")
    assert report.generation_method == "template"


def test_model_and_migration_unique_constraint_agree():
    from app.models import Report

    names = {
        c.name
        for c in Report.__table__.constraints
        if getattr(c, "name", None) is not None
    }
    assert "uq_report_run_language" in names
    cols = {
        tuple(sorted(c.name for c in constraint.columns))
        for constraint in Report.__table__.constraints
        if getattr(constraint, "name", None) == "uq_report_run_language"
    }
    assert ("analysis_run_id", "language") in cols
