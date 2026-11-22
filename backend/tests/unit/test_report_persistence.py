"""Report persistence: generated reports must be saved and reused."""

import uuid

from app.services.report_service import ReportService
from tests.unit.test_report_template import make_run


class RecordingRepo:
    def __init__(self, run, summary=None):
        self.run = run
        self.summary = summary
        self.saved = []
        self.by_lang = {}

    def get(self, analysis_id):
        return self.run

    def summary_of(self, run_id):
        return self.summary

    def report_of(self, run_id, language):
        return self.by_lang.get(language)

    def save_report(self, report):
        self.saved.append(report)
        report.created_at = report.created_at or __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        self.by_lang[report.language] = report
        return report


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


def test_bn_and_en_both_persist_separately():
    run, summary = make_run(dict())
    repo = RecordingRepo(run, summary)
    service = ReportService(repo=repo, llm=None)
    service.get_report(run.id, "bn")
    service.get_report(run.id, "en")
    assert {r.language for r in repo.saved} == {"bn", "en"}
